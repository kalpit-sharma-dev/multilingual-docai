# [DEPRECATED] PS-05 API Usage Guide: No-Annotation Evaluation

This document is deprecated. Please refer to:
- Main: `README.md`
- Evaluation Day: `docs/EVALUATION_DAY_RUNBOOK.md`
- Swagger (when running): http://localhost:8000/docs

## **Overview**
This guide covers using the PS-05 API for **evaluator testing scenarios** where:
- ✅ **No ground truth annotations are provided**
- ✅ **Large datasets (20GB+) are uploaded**
- ✅ **Predictions are generated for evaluation**
- ✅ **mAP calculation is not possible**

## **Quick Start for Evaluators**

### **1. Start the API**
```bash
# Using Docker Compose (Recommended)
docker-compose up -d

# Check status
curl http://localhost:8000/status
```

### **2. Upload Large Dataset (No Annotations)**
```bash
# Upload multiple images without annotations
curl -X POST "http://localhost:8000/upload-dataset" \
  -F "files=@document1.png" \
  -F "files=@document2.png" \
  -F "files=@document3.png" \
  -F "dataset_name=test_dataset_20gb"
```

**Response:**
```json
{
  "dataset_id": "uuid-here",
  "message": "Dataset uploaded successfully",
  "num_images": 3,
  "has_annotations": false,
  "total_size_gb": 0.05,
  "evaluation_mode": "prediction_only",
  "timestamp": "2024-01-01T12:00:00"
}
```

### **3. Process All Stages**
```bash
# Process all 3 stages in background
curl -X POST "http://localhost:8000/process-all" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid-here",
    "config": {
      "batch_size": 10,
      "high_quality": true
    }
  }'
```

**Response:**
```json
{
  "job_id": "job-uuid-here",
  "message": "Processing started in background",
  "status": "processing",
  "timestamp": "2024-01-01T12:00:00"
}
```

### **4. Get Predictions (Main Endpoint for Evaluators)**
```bash
# Get predictions for evaluation
curl "http://localhost:8000/predictions/uuid-here"
```

## **Complete Workflow Example (Python)**

### **Step 1: Upload Large Dataset**
```python
import requests
import os
from pathlib import Path

def upload_large_dataset(image_folder, dataset_name="test_dataset"):
    """Upload large dataset without annotations."""
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_folder).glob(f"*{ext}"))
    
    print(f"Found {len(image_files)} images")
    
    # Prepare files for upload
    files = []
    for img_path in image_files:
        files.append(('files', open(img_path, 'rb')))
    
    # Upload dataset
    response = requests.post(
        'http://localhost:8000/upload-dataset',
        files=files,
        data={'dataset_name': dataset_name}
    )
    
    if response.status_code == 200:
        dataset_info = response.json()
        print(f"Dataset uploaded: {dataset_info['dataset_id']}")
        print(f"Total size: {dataset_info['total_size_gb']:.2f} GB")
        return dataset_info['dataset_id']
    else:
        print(f"Upload failed: {response.text}")
        return None

# Usage
dataset_id = upload_large_dataset("path/to/20gb/dataset")
```

### **Step 2: Process All Stages**
```python
def process_dataset(dataset_id):
    """Process all 3 stages for the dataset."""
    
    response = requests.post(
        'http://localhost:8000/process-all',
        json={
            'dataset_id': dataset_id,
            'config': {
                'batch_size': 10,
                'high_quality': True,
                'memory_optimization': True
            }
        }
    )
    
    if response.status_code == 200:
        job_info = response.json()
        print(f"Processing started: {job_info['job_id']}")
        return job_info['job_id']
    else:
        print(f"Processing failed: {response.text}")
        return None

# Start processing
job_id = process_dataset(dataset_id)
```

### **Step 3: Monitor Progress and Get Results**
```python
import time

def wait_for_completion(dataset_id, max_wait_minutes=60):
    """Wait for processing to complete."""
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while time.time() - start_time < max_wait_seconds:
        try:
            # Try to get predictions
            response = requests.get(f'http://localhost:8000/predictions/{dataset_id}')
            
            if response.status_code == 200:
                print("Processing completed!")
                return response.json()
            elif response.status_code == 400 and "No results found" in response.text:
                print("Still processing... waiting 30 seconds")
                time.sleep(30)
            else:
                print(f"Unexpected response: {response.text}")
                time.sleep(30)
                
        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(30)
    
    print("Timeout waiting for completion")
    return None

# Wait for completion
results = wait_for_completion(dataset_id)
```

### **Step 4: Extract Predictions for Evaluation**
```python
def extract_predictions_for_evaluation(results):
    """Extract predictions in format suitable for evaluation."""
    
    predictions = results['predictions']
    
    # Stage 1: Layout Detection
    stage1_results = predictions.get('stage_1', {}).get('results', [])
    layout_predictions = []
    
    for element in stage1_results:
        layout_predictions.append({
            'type': element['type'],
            'bbox': [element['bbox']['x'], element['bbox']['y'], 
                    element['bbox']['width'], element['bbox']['height']],
            'confidence': element['confidence']
        })
    
    # Stage 2: Text Extraction
    stage2_results = predictions.get('stage_2', {}).get('results', [])
    text_predictions = []
    
    for element in stage2_results:
        text_predictions.append({
            'text': element['text'],
            'bbox': [element['bbox']['x'], element['bbox']['y'], 
                    element['bbox']['width'], element['bbox']['height']],
            'language': element['language'],
            'confidence': element['confidence']
        })
    
    # Stage 3: Content Understanding
    stage3_results = predictions.get('stage_3', {}).get('results', [])
    content_predictions = []
    
    for element in stage3_results:
        content_predictions.append({
            'type': element['type'],
            'description': element['description'],
            'bbox': [element['bbox']['x'], element['bbox']['y'], 
                    element['bbox']['width'], element['bbox']['height']],
            'confidence': element['confidence']
        })
    
    return {
        'layout_detection': layout_predictions,
        'text_extraction': text_predictions,
        'content_understanding': content_predictions,
        'total_elements': len(layout_predictions) + len(text_predictions) + len(content_predictions)
    }

# Extract predictions
evaluation_data = extract_predictions_for_evaluation(results)
print(f"Total elements detected: {evaluation_data['total_elements']}")
```

## **API Endpoints for No-Annotation Scenarios**

### **📤 Upload Dataset (No Annotations)**
```bash
POST /upload-dataset
```
- **Purpose**: Upload images without ground truth
- **Response**: Dataset ID and size information
- **Note**: Set `evaluation_mode` to "prediction_only"

### **🚀 Process All Stages**
```bash
POST /process-all
```
- **Purpose**: Process all 3 stages in background
- **Optimization**: Memory management for large datasets
- **Response**: Job ID for tracking

### **📊 Get Predictions (Main Endpoint)**
```bash
GET /predictions/{dataset_id}
```
- **Purpose**: Get predictions for evaluation
- **Format**: Structured JSON with all stage results
- **Note**: No mAP calculation possible

### **📋 Get Results**
```bash
GET /results/{dataset_id}
```
- **Purpose**: Get raw processing results
- **Format**: Detailed stage-by-stage results

## **Output Format for Evaluation**

### **Predictions Response Structure**
```json
{
  "dataset_id": "uuid-here",
  "dataset_name": "test_dataset_20gb",
  "num_images": 1000,
  "total_size_gb": 20.5,
  "predictions": {
    "stage_1": {
      "dataset_id": "uuid-here",
      "stage": "1",
      "status": "completed",
      "results": [
        {
          "type": "Text",
          "bbox": {
            "x": 100.0,
            "y": 200.0,
            "width": 300.0,
            "height": 50.0
          },
          "confidence": 0.95,
          "text": "",
          "language": null,
          "description": null
        }
      ],
      "processing_time": 45.2,
      "timestamp": "2024-01-01T12:00:00"
    },
    "stage_2": {
      "dataset_id": "uuid-here",
      "stage": "2",
      "status": "completed",
      "results": [
        {
          "type": "Text",
          "bbox": {
            "x": 100.0,
            "y": 200.0,
            "width": 300.0,
            "height": 50.0
          },
          "confidence": 0.92,
          "text": "Sample text content",
          "language": "en",
          "description": null
        }
      ],
      "processing_time": 120.5,
      "timestamp": "2024-01-01T12:00:00"
    },
    "stage_3": {
      "dataset_id": "uuid-here",
      "stage": "3",
      "status": "completed",
      "results": [
        {
          "type": "Table",
          "bbox": {
            "x": 400.0,
            "y": 300.0,
            "width": 200.0,
            "height": 150.0
          },
          "confidence": 0.88,
          "text": null,
          "language": null,
          "description": "Table showing data with 3 columns and 5 rows"
        }
      ],
      "processing_time": 85.3,
      "timestamp": "2024-01-01T12:00:00"
    }
  },
  "timestamp": "2024-01-01T12:00:00",
  "message": "Predictions generated successfully. No ground truth available for mAP calculation.",
  "note": "No ground truth annotations provided. Use predictions for evaluation."
}
```

## **Configuration for Large Datasets**

### **Memory Optimization Settings**
```json
{
  "batch_size": 10,
  "memory_optimization": true,
  "high_quality": true,
  "cleanup_frequency": "every_batch"
}
```

### **Stage-Specific Settings**
```json
{
  "stage_1": {
    "batch_size": 10,
    "confidence_threshold": 0.5
  },
  "stage_2": {
    "batch_size": 5,
    "ocr_engine": "easyocr",
    "languages": ["en", "hi", "ur", "ar", "ne", "fa"]
  },
  "stage_3": {
    "batch_size": 8,
    "use_advanced_models": true
  }
}
```

## **Performance & Scaling**

### **Large Dataset Handling (20GB+)**
- ✅ **Streaming uploads** to avoid memory issues
- ✅ **Batch processing** with configurable batch sizes
- ✅ **Memory cleanup** between batches and stages
- ✅ **Progress tracking** for long-running jobs
- ✅ **Background processing** with job IDs

### **Resource Requirements**
- **CPU**: 48 cores (as per challenge specs)
- **RAM**: 256GB with memory optimization
- **Storage**: SSD recommended for large datasets
- **GPU**: Optional for faster processing

### **Expected Processing Times**
- **Stage 1 (Layout)**: ~2-5 seconds per image
- **Stage 2 (Text)**: ~5-15 seconds per image
- **Stage 3 (Content)**: ~3-10 seconds per image
- **Total for 1000 images**: ~2-8 hours (depending on hardware)

## **Troubleshooting**

### **Common Issues**

1. **Memory Issues with Large Datasets**
   - Reduce batch size in config
   - Monitor memory usage via `/status`
   - Restart container if needed

2. **Processing Hangs**
   - Check logs: `docker logs ps05-document-ai`
   - Verify system resources
   - Use smaller batch sizes

3. **Upload Failures**
   - Check file formats (PNG, JPEG, TIFF supported)
   - Verify file permissions
   - Use streaming upload for large files

### **Monitoring & Debugging**
```bash
# Check system status
curl http://localhost:8000/status

# View container logs
docker logs ps05-document-ai

# Follow logs in real-time
docker logs -f ps05-document-ai

# Check memory usage
docker exec ps05-document-ai ps aux | grep python
```

## **Evaluation Notes**

### **Important Considerations**
1. **No mAP Calculation**: Without ground truth, mAP cannot be computed
2. **Prediction Quality**: Focus on prediction confidence scores
3. **Output Consistency**: Verify all stages produce consistent results
4. **Processing Completeness**: Ensure all images are processed

### **Evaluation Metrics Available**
- ✅ **Detection counts** per stage
- ✅ **Confidence scores** for predictions
- ✅ **Processing times** per stage
- ✅ **Element types** detected
- ✅ **Text content** extracted
- ✅ **Language identification** results

### **What Evaluators Can Assess**
1. **Model Performance**: Confidence scores and detection patterns
2. **Processing Speed**: Time per image and stage
3. **Output Quality**: Consistency and completeness
4. **System Stability**: Memory usage and error handling
5. **Scalability**: Performance with large datasets

## **Support & Contact**

For issues with large dataset processing:
1. Check memory usage and system resources
2. Reduce batch sizes if memory issues occur
3. Monitor logs for error patterns
4. Verify dataset format and file integrity

The API is designed to handle large datasets efficiently while providing comprehensive predictions for evaluation.
