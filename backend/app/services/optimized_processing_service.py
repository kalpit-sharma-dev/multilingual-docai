#!/usr/bin/env python3
"""
Optimized Processing Service for PS-05 Challenge
Optimized for A100 GPU with 2-hour evaluation time limit

Fully compliant with PS-05 requirements:
- Preprocessing: De-skew, denoise, augmentation
- Layout Detection: 6 classes (Background, Text, Title, List, Table, Figure)
- Text Extraction: Multilingual OCR with language ID
- Content Understanding: Natural language descriptions
- Output: JSON per image with bounding boxes and content
"""

import asyncio
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from dataclasses import dataclass
from ultralytics import YOLO
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import easyocr
import fasttext
from PIL import Image
import cv2
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for optimized processing."""
    batch_size: int = 50  # Optimized for A100 memory
    max_workers: int = 8  # Parallel processing
    gpu_acceleration: bool = True
    memory_optimization: bool = True
    output_format: str = "json"
    save_intermediate: bool = True
    enable_preprocessing: bool = True  # Enable de-skew, denoise, augmentation
    enable_augmentation: bool = True  # Enable training augmentation

@dataclass
class ProcessingResult:
    """Result from processing a single image."""
    filename: str
    elements: List[Dict[str, Any]]
    processing_time: float
    stage_results: Dict[str, Any]
    gpu_memory_used: Optional[float] = None

class OptimizedProcessingService:
    """
    High-performance processing service optimized for A100 GPU.
    Processes all 3 stages in parallel for maximum speed.
    Fully compliant with PS-05 requirements.
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.device = self._setup_device()
        self.models = self._load_models()
        self.results_cache = {}
        
    def _setup_device(self) -> str:
        """Setup GPU device with optimization."""
        if self.config.gpu_acceleration and torch.cuda.is_available():
            device = "cuda"
            # Optimize CUDA settings for A100
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction for optimal usage
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.9)
                logger.info(f"GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            logger.info("Using CPU processing")
        
        return device
    
    def _load_models(self) -> Dict[str, Any]:
        """Load all models with GPU optimization."""
        models = {}
        
        try:
            # Stage 1: Layout Detection (GPU optimized)
            logger.info("Loading YOLO for 6-class layout detection...")
            try:
                # Prefer fine-tuned 6-class weights if present; else use env; else fallback to yolov8x
                default_layout_weights = "/app/models/layout_yolo_6class.pt"
                env_weights = os.environ.get("YOLO_WEIGHTS")
                yolo_weights = default_layout_weights if os.path.exists(default_layout_weights) else (env_weights or "/app/models/yolov8x.pt")
                models['yolo'] = YOLO(yolo_weights)
                if self.device == "cuda":
                    models['yolo'].to(self.device)
            except Exception as e:
                logger.warning(f"YOLO weights not found or failed to load ({e}); using default 'yolov8x' from hub.")
                models['yolo'] = YOLO('yolov8x.pt')
                if self.device == "cuda":
                    try:
                        models['yolo'].to(self.device)
                    except Exception:
                        pass
            
            # LayoutLMv3 for fine-grained classification (optional)
            try:
                logger.info("Loading LayoutLMv3 (optional)...")
                layoutlm_ckpt = os.environ.get("LAYOUTLMV3_CHECKPOINT", "")
                if layoutlm_ckpt:
                    models['layoutlmv3'] = LayoutLMv3ForSequenceClassification.from_pretrained(layoutlm_ckpt)
                    models['layoutlmv3_processor'] = LayoutLMv3Processor.from_pretrained(layoutlm_ckpt)
                    if self.device == "cuda":
                        models['layoutlmv3'].to(self.device)
                else:
                    logger.info("LAYOUTLMV3_CHECKPOINT not set; skipping LayoutLMv3 refinement.")
            except Exception as e:
                logger.warning(f"LayoutLMv3 unavailable; skipping refinement: {e}")
            
            # Stage 2: OCR and Language Detection
            # If PaddleOCR is enabled, skip EasyOCR loading to avoid offline model dependency
            use_paddle = os.environ.get("USE_PADDLEOCR", "0") == "1"
            easyocr_dir = os.environ.get("EASYOCR_MODEL_PATH", "/app/models/easyocr")
            if use_paddle:
                models['ocr'] = None
                logger.info("Skipping EasyOCR load because USE_PADDLEOCR=1")
            else:
                logger.info("Loading EasyOCR...")
                try:
                    models['ocr'] = easyocr.Reader(
                        ['en', 'hi', 'ur', 'ar', 'ne', 'fa'],
                        gpu=(self.device=="cuda"),
                        model_storage_directory=easyocr_dir,
                        download_enabled=False
                    )
                except Exception as e:
                    logger.warning(f"EasyOCR unavailable; will continue without it: {e}")
                    models['ocr'] = None
            # Optional: PaddleOCR as primary (enabled when USE_PADDLEOCR=1)
            try:
                if use_paddle:
                    from paddleocr import PaddleOCR  # type: ignore
                    models['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=(self.device=="cuda"))
                    logger.info("PaddleOCR enabled as primary OCR (lang='en').")
            except Exception as e:
                logger.warning(f"PaddleOCR not enabled: {e}")
            
            # Language detection (optional fastText)
            try:
                logger.info("Loading fastText language detection...")
                lid_path = Path('lid.176.bin')
                if not lid_path.exists():
                    alt_path = Path('/app/lid.176.bin')
                    if alt_path.exists():
                        lid_path = alt_path
                if lid_path.exists():
                    models['langdetect'] = fasttext.load_model(str(lid_path))
                else:
                    logger.warning("lid.176.bin not found; language detection will default to 'en'.")
                    models['langdetect'] = None
            except Exception as e:
                logger.warning(f"fastText language model unavailable; defaulting language to 'en': {e}")
                models['langdetect'] = None
            
            # Stage 3: Content Understanding (optional)
            try:
                logger.info("Loading BLIP-2 for image captioning (optional)...")
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                blip2_ckpt = os.environ.get("BLIP2_CHECKPOINT", "")
                if blip2_ckpt:
                    models['blip2_processor'] = Blip2Processor.from_pretrained(blip2_ckpt)
                    models['blip2'] = Blip2ForConditionalGeneration.from_pretrained(blip2_ckpt)
                    if self.device == "cuda":
                        models['blip2'].to(self.device)
                else:
                    logger.info("BLIP2_CHECKPOINT not set; skipping content captioning.")
            except Exception as e:
                logger.warning(f"BLIP-2 unavailable; skipping content captioning: {e}")
            
            logger.info("Core models loaded (with graceful fallbacks).")

            # Optional specialized captioner for charts (Pix2Struct)
            try:
                chart_ckpt = os.environ.get("CHART_CAPTION_CHECKPOINT", "")
                if chart_ckpt:
                    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
                    models['chart_processor'] = Pix2StructProcessor.from_pretrained(chart_ckpt)
                    models['chart_model'] = Pix2StructForConditionalGeneration.from_pretrained(chart_ckpt)
                    if self.device == "cuda":
                        models['chart_model'].to(self.device)
                    logger.info(f"Chart captioning enabled: {chart_ckpt}")
            except Exception as e:
                logger.warning(f"Chart captioner not enabled: {e}")

            # Optional table-to-text using seq2seq LM on OCR text context
            try:
                table_ckpt = os.environ.get("TABLE_T2T_CHECKPOINT", "")
                if table_ckpt:
                    models['table_tokenizer'] = AutoTokenizer.from_pretrained(table_ckpt)
                    models['table_model'] = AutoModelForSeq2SeqLM.from_pretrained(table_ckpt)
                    if self.device == "cuda":
                        models['table_model'].to(self.device)
                    logger.info(f"Table-to-text enabled: {table_ckpt}")
            except Exception as e:
                logger.warning(f"Table-to-text not enabled: {e}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
        
        return models

    def _normalize_ps05_label(self, raw_label: str, class_id: Optional[int] = None, total_classes: Optional[int] = None) -> str:
        """Normalize diverse model label names to PS-05 canonical labels.
        Canonical: Background, Text, Title, List, Table, Figure.
        """
        try:
            ps05_labels = ['Background', 'Text', 'Title', 'List', 'Table', 'Figure']
            if total_classes == 6 and class_id is not None:
                return ps05_labels[min(class_id, 5)]
            name = (raw_label or '').strip().lower()
            synonyms = {
                'background': 'Background', 'bg': 'Background', 'other': 'Background',
                'text': 'Text', 'paragraph': 'Text', 'body': 'Text', 'content': 'Text',
                'title': 'Title', 'heading': 'Title', 'header': 'Title', 'headline': 'Title',
                'list': 'List', 'bullet': 'List', 'enumeration': 'List',
                'table': 'Table', 'tabular': 'Table',
                'figure': 'Figure', 'image': 'Figure', 'chart': 'Figure', 'diagram': 'Figure', 'photo': 'Figure'
            }
            return synonyms.get(name, raw_label)
        except Exception:
            return raw_label

    def _convert_quad_to_hbb_xywh(self, quad: Any) -> List[int]:
        """Convert a 4-point quadrilateral to HBB in [x, y, w, h] order.
        EasyOCR returns [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
        """
        try:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            min_x, max_x = int(min(xs)), int(max(xs))
            min_y, max_y = int(min(ys)), int(max(ys))
            w = max_x - min_x
            h = max_y - min_y
            if w < 0: w = 0
            if h < 0: h = 0
            return [min_x, min_y, w, h]
        except Exception:
            # Fallback safe box
            return [0, 0, 0, 0]

    def _caption_from_pil(self, pil_image: Image.Image) -> str:
        """Generate a caption using BLIP-2 for a given PIL image region."""
        try:
            if not self.models.get('blip2_processor') or not self.models.get('blip2'):
                return ""
            inputs = self.models['blip2_processor'](pil_image, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.models['blip2'].generate(**inputs, max_length=100)
                description = self.models['blip2_processor'].decode(outputs[0], skip_special_tokens=True)
            return description
        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            return ""

    def _caption_chart_from_pil(self, pil_image: Image.Image) -> str:
        """Caption charts using Pix2Struct if available; fallback to BLIP-2."""
        if self.models.get('chart_model') and self.models.get('chart_processor'):
            try:
                processor = self.models['chart_processor']
                model = self.models['chart_model']
                inputs = processor(images=pil_image, text="Generate a detailed chart description.", return_tensors="pt")
                if self.device == 'cuda':
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=128)
                return processor.decode(output_ids[0], skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Pix2Struct caption failed, fallback to BLIP-2: {e}")
        return self._caption_from_pil(pil_image)

    def _table_to_text(self, table_text: str) -> str:
        """Summarize table OCR text using seq2seq if available; fallback to BLIP-2 will be used on image crop elsewhere."""
        if not table_text.strip():
            return ""
        if self.models.get('table_model') and self.models.get('table_tokenizer'):
            try:
                tokenizer = self.models['table_tokenizer']
                model = self.models['table_model']
                prompt = f"summarize table: {table_text}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                if self.device == 'cuda':
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=128)
                return tokenizer.decode(output_ids[0], skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Table T2T generation failed: {e}")
        return ""

    def _collect_text_in_region(self, region_xywh: List[int], text_elements: List[Dict[str, Any]]) -> str:
        """Collect OCR text within a region bbox [x,y,w,h] and concatenate in reading order."""
        x, y, w, h = region_xywh
        lines = []
        for te in text_elements:
            bx = te.get('bbox', [0, 0, 0, 0])
            tx, ty, tw, th = bx
            if tx >= x and ty >= y and tx + tw <= x + w and ty + th <= y + h:
                if te.get('text'):
                    lines.append(te['text'])
        return ' '.join(lines[:200])  # cap to avoid overly long prompts

    def _sort_text_elements_reading_order(self, text_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort text elements in reading order; handle RTL languages heuristically."""
        try:
            # Group by rows (y), then sort x; reverse x for RTL languages
            # Detect predominant language from elements (simple vote)
            langs = [te.get('language') for te in text_elements if te.get('language')]
            rtl_langs = {'ar', 'fa', 'ur'}
            is_rtl = any(l in rtl_langs for l in langs)
            ordered = sorted(text_elements, key=lambda e: (e.get('bbox', [0, 0, 0, 0])[1], e.get('bbox', [0, 0, 0, 0])[0]))
            if is_rtl:
                # Within same row (approx y), reverse x order
                row_threshold = 10
                rows: List[List[Dict[str, Any]]] = []
                for el in ordered:
                    placed = False
                    for row in rows:
                        if abs(row[0]['bbox'][1] - el['bbox'][1]) <= row_threshold:
                            row.append(el)
                            placed = True
                            break
                    if not placed:
                        rows.append([el])
                reordered: List[Dict[str, Any]] = []
                for row in rows:
                    row_sorted = sorted(row, key=lambda e: e['bbox'][0], reverse=True)
                    reordered.extend(row_sorted)
                ordered = reordered
            for idx, el in enumerate(ordered):
                el['order'] = idx
            return ordered
        except Exception:
            return text_elements

    def _refine_layout_with_layoutlm(self, image_bgr: np.ndarray, layout_elements: List[Dict[str, Any]], text_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optionally refine YOLO labels using LayoutLMv3 if a 6-class head is available.
        This is safe: if num_labels != 6 or processor/model missing, returns original labels.
        """
        try:
            model = self.models.get('layoutlmv3')
            processor = self.models.get('layoutlmv3_processor')
            if model is None or processor is None:
                return layout_elements
            if getattr(model.config, 'num_labels', 0) != 6:
                # Not a 6-class head -> skip refinement
                return layout_elements
            id2label = getattr(model.config, 'id2label', {i: l for i, l in enumerate(['Background','Text','Title','List','Table','Figure'])})

            h_img, w_img = image_bgr.shape[:2]
            pil_full = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

            refined = []
            for el in layout_elements:
                x, y, h, w = el.get('bbox', [0, 0, 0, 0])
                # bbox is [x,y,w,h]
                x, y, w, h = el.get('bbox', [0, 0, 0, 0])
                crop = image_bgr[y:y+h, x:x+w]
                if crop is None or crop.size == 0:
                    refined.append(el)
                    continue
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                # Collect OCR words inside this region
                words, boxes = [], []
                for te in text_elements:
                    bx = te.get('bbox', [0, 0, 0, 0])
                    tx, ty, tw, th = bx
                    if tx >= x and ty >= y and tx + tw <= x + w and ty + th <= y + h:
                        if te.get('text'):
                            words.append(te['text'])
                            # Normalize to 0-1000 per LayoutLM convention
                            bxn = int(((tx - x) / max(w, 1)) * 1000)
                            byn = int(((ty - y) / max(h, 1)) * 1000)
                            bwn = int((tw / max(w, 1)) * 1000)
                            bhn = int((th / max(h, 1)) * 1000)
                            boxes.append([bxn, byn, bxn + bwn, byn + bhn])

                if not words:
                    refined.append(el)
                    continue

                encoded = processor(images=pil_crop, words=[words], boxes=[boxes], return_tensors="pt", truncation=True, padding=True)
                if self.device == 'cuda':
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.no_grad():
                    logits = model(**encoded).logits
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    conf, pred = torch.max(probs, dim=-1)
                    conf_val = float(conf.item())
                    pred_label = id2label.get(int(pred.item()), el.get('type'))
                # Only override if confident
                if conf_val >= 0.8:
                    el_ref = dict(el)
                    el_ref['type'] = pred_label
                    el_ref['refined_by'] = 'layoutlmv3'
                    el_ref['refine_confidence'] = conf_val
                    refined.append(el_ref)
                else:
                    refined.append(el)

            return refined
        except Exception as e:
            logger.warning(f"LayoutLMv3 refinement skipped: {e}")
            return layout_elements
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing: De-skew, denoise, and normalize image.
        Implements requirements from PS-05 document.
        """
        if not self.config.enable_preprocessing:
            return image
        
        try:
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 1. De-skew using Hough transform
            coords = np.column_stack(np.where(gray > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = 90 + angle
                if angle != 0:
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # 2. Denoise using Non-local Means
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # 3. Normalize and enhance contrast
            normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert back to BGR if original was color
            if len(image.shape) == 3:
                result = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            else:
                result = normalized
                
            logger.info("Image preprocessing completed: de-skew, denoise, normalize")
            return result
            
        except Exception as e:
            logger.warning(f"Preprocessing failed, using original image: {e}")
            return image
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Image augmentation for training robustness.
        Implements requirements from PS-05 document.
        """
        if not self.config.enable_augmentation:
            return image
        
        try:
            # Random blur for robustness
            if np.random.random() < 0.3:
                kernel_size = np.random.choice([3, 5, 7])
                image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            # Random rotation for robustness
            if np.random.random() < 0.3:
                angle = np.random.uniform(-15, 15)
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # Random noise for robustness
            if np.random.random() < 0.2:
                noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
                image = cv2.add(image, noise)
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            logger.info("Image augmentation completed: blur, rotation, noise")
            return image
            
        except Exception as e:
            logger.warning(f"Augmentation failed, using original image: {e}")
            return image

    async def process_dataset_parallel(self, dataset_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process entire dataset with maximum speed optimization.
        All stages run in parallel for optimal performance.
        """
        start_time = time.time()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of images
        image_files = self._get_image_files(dataset_path)
        total_images = len(image_files)
        
        logger.info(f"Processing {total_images} images with parallel optimization...")
        
        # Process in optimized batches
        results = []
        batch_size = self.config.batch_size
        
        for batch_idx in range(0, total_images, batch_size):
            batch_files = image_files[batch_idx:batch_idx + batch_size]
            logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
            
            # Process batch with all stages in parallel
            batch_results = await self._process_batch_parallel(batch_files, output_path)
            results.extend(batch_results)
            
            # Save intermediate results
            if self.config.save_intermediate:
                self._save_batch_results(batch_results, output_path, batch_idx)
        
        total_time = time.time() - start_time
        processing_speed = total_images / total_time
        
        logger.info(f"Processing completed in {total_time:.2f}s")
        logger.info(f"Speed: {processing_speed:.2f} images/second")
        
        return {
            "total_images": total_images,
            "processing_time": total_time,
            "speed_images_per_second": processing_speed,
            "results": results,
            "output_directory": str(output_path)
        }
    
    async def _process_batch_parallel(self, image_files: List[str], output_path: Path) -> List[ProcessingResult]:
        """
        Process a batch of images with all stages running in parallel.
        This is the core optimization for maximum speed.
        """
        # Create tasks for all stages simultaneously
        tasks = []
        
        for image_file in image_files:
            # Stage 1: Layout Detection
            task1 = asyncio.create_task(self._detect_layout_gpu(image_file))
            
            # Stage 2: Text Extraction + Language ID
            task2 = asyncio.create_task(self._extract_text_and_language(image_file))
            
            # Stage 3: Content Understanding
            task3 = asyncio.create_task(self._understand_content_gpu(image_file))
            
            # Combine all tasks
            combined_task = asyncio.create_task(self._combine_stage_results(
                image_file, task1, task2, task3
            ))
            tasks.append(combined_task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results
    
    async def _detect_layout_gpu(self, image_file: str) -> Dict[str, Any]:
        """Stage 1: Fast layout detection using GPU-optimized YOLOv8x."""
        try:
            # Load and preprocess image
            image = cv2.imread(image_file)
            if image is None:
                raise ValueError(f"Could not load image: {image_file}")
            
            # Apply preprocessing
            image = self._preprocess_image(image)
            
            # Convert to PIL for YOLO
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Run YOLO inference on GPU with tuned thresholds (class-aware NMS)
            results = self.models['yolo'](
                pil_image,
                device=self.device,
                conf=0.35,
                iou=0.50,
                imgsz=1024,
                agnostic_nms=False
            )
            
            # Process results
            layout_elements = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        # Determine label name from model metadata if available
                        label = str(class_id)
                        try:
                            # Ultralytics results carry a names dict
                            names = getattr(result, 'names', None)
                            if isinstance(names, dict):
                                label = names.get(class_id, label)
                            else:
                                # Try model names
                                model_names = getattr(self.models['yolo'].model, 'names', None)
                                if isinstance(model_names, dict):
                                    label = model_names.get(class_id, label)
                        except Exception:
                            pass
                        # Normalize label to PS-05 canonical labels
                        model_num_classes = None
                        try:
                            model_names = getattr(self.models['yolo'].model, 'names', None)
                            if isinstance(model_names, dict):
                                model_num_classes = len(model_names)
                        except Exception:
                            model_num_classes = None
                        label = self._normalize_ps05_label(label, class_id, model_num_classes)
                        # Skip background detections
                        if label == 'Background':
                            continue
                        
                        # Standardize to [x, y, w, h]
                        layout_elements.append({
                            "type": label,
                            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                            "confidence": float(confidence),
                            "class_id": class_id
                        })
            
            return {"layout_elements": layout_elements}
            
        except Exception as e:
            logger.error(f"Error in layout detection for {image_file}: {e}")
            return {"layout_elements": [], "error": str(e)}
    
    async def _extract_text_and_language(self, image_file: str) -> Dict[str, Any]:
        """Stage 2: Text extraction and language identification."""
        try:
            # Load and preprocess image
            image = cv2.imread(image_file)
            if image is None:
                raise ValueError(f"Could not load image: {image_file}")
            
            # Apply preprocessing
            image = self._preprocess_image(image)
            
            # OCR: Prefer PaddleOCR if enabled, fallback to EasyOCR
            ocr_results = []
            try:
                if self.models.get('paddleocr') and os.environ.get("USE_PADDLEOCR", "0") == "1":
                    paddle_out = self.models['paddleocr'].ocr(image, cls=True)
                    # paddle_out is list per image; we pass single image
                    if isinstance(paddle_out, list) and len(paddle_out) > 0:
                        for line in paddle_out[0]:
                            try:
                                quad = line[0]
                                text = line[1][0]
                                score = float(line[1][1])
                                # Normalize quad to [[x,y],...]
                                if isinstance(quad, list) and len(quad) >= 4 and isinstance(quad[0], (list, tuple)):
                                    bbox_quad = [[int(pt[0]), int(pt[1])] for pt in quad[:4]]
                                    ocr_results.append((bbox_quad, text, score))
                            except Exception:
                                continue
            except Exception as e:
                logger.warning(f"PaddleOCR failed, will fallback to EasyOCR: {e}")
                ocr_results = []
            if not ocr_results:
                # EasyOCR (multilingual support as per requirements)
                ocr_results = self.models['ocr'].readtext(image)
            
            text_elements = []
            for (bbox, text, confidence) in ocr_results:
                # Detect language using fastText (176 languages as per requirements)
                if text.strip():
                    if self.models.get('langdetect') is not None:
                        lang_pred = self.models['langdetect'].predict(text, k=1)
                        language = lang_pred[0][0].replace('__label__', '')
                    else:
                        language = 'en'
                    
                    # Ensure language is one of the target languages
                    target_languages = ['en', 'hi', 'ur', 'ar', 'ne', 'fa']
                    if language not in target_languages:
                        # Map to closest target language
                        language = 'en'  # Default to English
                    
                    # Convert EasyOCR quad bbox to HBB [x, y, w, h]
                    hbb_xywh = self._convert_quad_to_hbb_xywh(bbox)
                    text_elements.append({
                        "text": text,
                        "bbox": hbb_xywh,
                        "confidence": confidence,
                        "language": language
                    })
            
            # Order text in reading sequence
            text_elements = self._sort_text_elements_reading_order(text_elements)
            return {"text_elements": text_elements}
            
        except Exception as e:
            logger.error(f"Error in text extraction for {image_file}: {e}")
            return {"text_elements": [], "error": str(e)}
    
    async def _understand_content_gpu(self, image_file: str) -> Dict[str, Any]:
        """Stage 3: Content understanding using GPU-optimized models."""
        try:
            # Load image
            image = cv2.imread(image_file)
            if image is None:
                raise ValueError(f"Could not load image: {image_file}")
            
            # Apply preprocessing
            image = self._preprocess_image(image)
            
            # Convert to PIL for BLIP-2
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Generate description using BLIP-2 (if available)
            if not self.models.get('blip2_processor') or not self.models.get('blip2'):
                return {"description": ""}
            inputs = self.models['blip2_processor'](pil_image, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.models['blip2'].generate(**inputs, max_length=100)
                description = self.models['blip2_processor'].decode(outputs[0], skip_special_tokens=True)
            
            return {"description": description}
            
        except Exception as e:
            logger.error(f"Error in content understanding for {image_file}: {e}")
            return {"description": "Error processing image", "error": str(e)}
    
    async def _combine_stage_results(self, image_file: str, task1, task2, task3) -> ProcessingResult:
        """Combine results from all three stages."""
        start_time = time.time()
        
        # Wait for all stages to complete
        stage1_result = await task1
        stage2_result = await task2
        stage3_result = await task3
        
        # Combine into final result
        filename = Path(image_file).name
        elements = []
        
        # Add layout elements (Stage 1)
        stage1_layout = stage1_result.get("layout_elements", [])
        stage2_text = stage2_result.get("text_elements", [])

        # Optional refinement using LayoutLMv3 if fine-tuned 6-class head is available
        try:
            base_img = cv2.imread(image_file)
            if base_img is not None:
                # Align refinement to deskewed space used by detection/OCR
                base_img = self._preprocess_image(base_img)
                stage1_layout = self._refine_layout_with_layoutlm(base_img, stage1_layout, stage2_text)
        except Exception as e:
            logger.warning(f"Layout refinement error: {e}")

        for element in stage1_layout:
            elements.append({
                "type": element["type"],
                "bbox": element["bbox"],  # standardized [x, y, h, w]
                "confidence": element["confidence"]
            })
        
        # Add text elements (Stage 2)
        for element in stage2_text:
            elements.append({
                "type": "Text",
                "bbox": element["bbox"],
                "content": element["text"],
                "language": element["language"],
                "confidence": element["confidence"]
            })
        
        # Add per-element captions for Table and Figure using BLIP-2 on cropped regions
        try:
            base_image = cv2.imread(image_file)
            if base_image is not None:
                # Align caption crops to deskewed coordinate space
                base_image = self._preprocess_image(base_image)
                for el in elements:
                    if el.get("type") in ["Table", "Figure"] and isinstance(el.get("bbox"), list):
                        x, y, w, h = el["bbox"]
                        if w > 0 and h > 0:
                            crop = base_image[y:y+h, x:x+w]
                            if crop is not None and crop.size > 0:
                                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                # Specialized paths + subtype heuristic for charts/maps
                                desc = ""
                                if el.get("type") == "Table":
                                    table_text = self._collect_text_in_region(el["bbox"], stage2_text)
                                    desc = self._table_to_text(table_text)
                                    if not desc:
                                        desc = self._caption_from_pil(pil_crop)
                                else:
                                    # Simple heuristic: if many straight lines/axes present, consider Chart; else Map/Image
                                    try:
                                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                        edges = cv2.Canny(gray, 50, 150)
                                        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=int(min(w, h)*0.4), maxLineGap=10)
                                        subtype = 'Chart' if lines is not None and len(lines) > 2 else 'MapOrImage'
                                    except Exception:
                                        subtype = 'Figure'
                                    el['subtype'] = subtype
                                    # Try chart captioner; fallback to BLIP-2
                                    desc = self._caption_chart_from_pil(pil_crop)
                                if desc:
                                    el["description"] = desc
        except Exception as e:
            logger.warning(f"Per-element captioning failed: {e}")

        # Optionally keep whole-image content description
        if stage3_result.get("description"):
            elements.append({
                "type": "Content",
                "description": stage3_result["description"]
            })
        
        processing_time = time.time() - start_time
        
        # Get GPU memory usage if available
        gpu_memory = None
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
        
        return ProcessingResult(
            filename=filename,
            elements=elements,
            processing_time=processing_time,
            stage_results={
                "stage1": stage1_result,
                "stage2": stage2_result,
                "stage3": stage3_result
            },
            gpu_memory_used=gpu_memory
        )
    
    def _get_image_files(self, dataset_path: str) -> List[str]:
        """Get list of image files from dataset path."""
        path = Path(dataset_path)
        if path.is_file():
            return [str(path)]
        elif path.is_dir():
            # Support all image formats as per PS-05 requirements
            image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.tif'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(path.glob(f"*{ext}"))
                image_files.extend(path.glob(f"*{ext.upper()}"))
            return [str(f) for f in sorted(image_files)]
        else:
            raise ValueError(f"Invalid dataset path: {dataset_path}")
    
    def _save_batch_results(self, batch_results: List[ProcessingResult], output_path: Path, batch_idx: int):
        """Save batch results to JSON files."""
        batch_dir = output_path / f"batch_{batch_idx//self.config.batch_size:03d}"
        batch_dir.mkdir(exist_ok=True)
        
        for result in batch_results:
            output_file = batch_dir / f"{result.filename}.json"
            
            # Convert to JSON-serializable format (exactly as per PS-05 requirements)
            json_data = {
                "filename": result.filename,
                "elements": result.elements,
                "processing_time": result.processing_time,
                "gpu_memory_used": result.gpu_memory_used,
                "timestamp": time.time()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics and GPU usage."""
        stats = {
            "device": self.device,
            "models_loaded": list(self.models.keys()),
            "config": self.config.__dict__
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            stats.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_cached_gb": torch.cuda.memory_reserved() / 1e9
            })
        
        return stats
