# 🚀 PS-05 Document Understanding API Usage Guide

## **Overview**
This API provides a complete 3-stage document understanding pipeline:
- **Stage 1**: Layout Detection (mAP evaluation)
- **Stage 2**: Text Extraction + Language ID
- **Stage 3**: Content Understanding + Natural Language

## **Quick Start**

### **1. Start the API**
```bash
# Using Docker Compose (Recommended)
docker-compose up -d

# Or build and run manually
docker build -t ps05-backend .
docker run -p 8000:8000 -v $(pwd)/data:/app/data ps05-backend
```

### **2. Check API Status**
```bash
curl http://localhost:8000/status
```

## **API Endpoints**

### **📤 Upload Dataset**
```bash
curl -X POST "http://localhost:8000/upload-dataset" \
  -F "files=@document1.png" \
  -F "files=@document2.png" \
  -F "annotations=@ground_truth.json"
```

**Response:**
```json
{
  "dataset_id": "uuid-here",
  "message": "Dataset uploaded successfully",
  "num_images": 2,
  "has_annotations": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

### **🔧 Process Single Stage**
```bash
curl -X POST "http://localhost:8000/process-stage" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid-here",
    "stage": "1",
    "config": {
      "confidence_threshold": 0.5,
      "iou_threshold": 0.5
    }
  }'
```

**Stages:**
- `"1"`: Layout Detection
- `"2"`: Text Extraction + Language ID  
- `"3"`: Content Understanding

### **🚀 Process All Stages**
```bash
curl -X POST "http://localhost:8000/process-all" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid-here",
    "config": {
      "high_quality": true,
      "use_advanced_models": true
    }
  }'
```

### **📊 Evaluate Results**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "uuid-here"}'
```

**Response includes:**
- mAP scores for layout detection
- CER/WER for text extraction
- Language identification accuracy
- Overall performance score
- Improvement recommendations

### **📋 Get Results**
```bash
curl "http://localhost:8000/results/uuid-here"
```

### **📚 List Datasets**
```bash
curl "http://localhost:8000/datasets"
```

## **Complete Workflow Example**

### **Step 1: Upload Dataset**
```python
import requests

# Upload images and annotations
files = [
    ('files', open('doc1.png', 'rb')),
    ('files', open('doc2.png', 'rb')),
    ('annotations', open('ground_truth.json', 'rb'))
]

response = requests.post(
    'http://localhost:8000/upload-dataset',
    files=files
)

dataset_id = response.json()['dataset_id']
print(f"Dataset uploaded: {dataset_id}")
```

### **Step 2: Process All Stages**
```python
# Process all stages
response = requests.post(
    'http://localhost:8000/process-all',
    json={'dataset_id': dataset_id}
)

job_id = response.json()['job_id']
print(f"Processing started: {job_id}")
```

### **Step 3: Evaluate Results**
```python
# Wait for processing to complete, then evaluate
response = requests.post(
    'http://localhost:8000/evaluate',
    json={'dataset_id': dataset_id}
)

evaluation = response.json()
print(f"mAP Score: {evaluation['stages']['stage_1']['mAP']:.3f}")
print(f"Overall Score: {evaluation['overall_score']:.3f}")
```

### **Step 4: Get Results**
```python
# Get all results
response = requests.get(f'http://localhost:8000/results/{dataset_id}')
results = response.json()

# Access stage results
stage1_results = results['results']['stage_1_results']
stage2_results = results['results']['stage_2_results']
stage3_results = results['results']['stage_3_results']
```

## **Input/Output Formats**

### **Ground Truth Annotations Format**
```json
{
  "annotations": [
    {
      "bbox": [100, 200, 300, 100],
      "category_id": 1,
      "category_name": "Text",
      "text": "Sample text content"
    },
    {
      "bbox": [400, 300, 200, 150],
      "category_id": 4,
      "category_name": "Table"
    }
  ]
}
```

### **Output JSON Format**
```json
{
  "filename": "document.png",
  "elements": [
    {
      "type": "Text",
      "bbox": {
        "x": 100,
        "y": 200,
        "width": 300,
        "height": 100
      },
      "confidence": 0.95,
      "text": "Sample text content",
      "language": "en",
      "description": null
    },
    {
      "type": "Table",
      "bbox": {
        "x": 400,
        "y": 300,
        "width": 200,
        "height": 150
      },
      "confidence": 0.88,
      "text": null,
      "language": null,
      "description": "Table showing data with 3 columns and 5 rows"
    }
  ]
}
```

## **Configuration Options**

### **Stage 1: Layout Detection**
```json
{
  "confidence_threshold": 0.5,
  "iou_threshold": 0.5,
  "model": "yolov8x",
  "image_size": 640
}
```

### **Stage 2: Text Extraction**
```json
{
  "ocr_engine": "easyocr",
  "languages": ["en", "hi", "ur", "ar", "ne", "fa"],
  "high_quality": true
}
```

### **Stage 3: Content Understanding**
```json
{
  "use_advanced_models": true,
  "description_length": "detailed",
  "include_metadata": true
}
```

## **Error Handling**

### **Common Error Responses**
```json
{
  "detail": "Dataset not found",
  "status_code": 404
}
```

```json
{
  "detail": "No ground truth annotations found for evaluation",
  "status_code": 400
}
```

### **HTTP Status Codes**
- `200`: Success
- `400`: Bad Request (missing annotations, invalid stage)
- `404`: Not Found (dataset, results)
- `500`: Internal Server Error

## **Performance & Scaling**

### **Resource Requirements**
- **CPU**: 48 cores (as per challenge specs)
- **RAM**: 256GB
- **GPU**: A100 (optional, for faster processing)
- **Storage**: SSD recommended for large datasets

### **Batch Processing**
For large datasets, use the `/process-all` endpoint which processes stages sequentially in the background.

### **Monitoring**
Check `/status` endpoint for:
- Available models
- Storage statistics
- System health

## **Troubleshooting**

### **Common Issues**

1. **Dataset Upload Fails**
   - Check file formats (PNG, JPEG supported)
   - Ensure annotations JSON is valid
   - Verify file permissions

2. **Processing Hangs**
   - Check system resources
   - Monitor logs: `docker logs ps05-document-ai`
   - Restart container if needed

3. **Low mAP Scores**
   - Verify ground truth format
   - Check annotation quality
   - Adjust confidence thresholds

### **Logs & Debugging**
```bash
# View container logs
docker logs ps05-document-ai

# Follow logs in real-time
docker logs -f ps05-document-ai

# Access container shell
docker exec -it ps05-document-ai bash
```

## **Advanced Usage**

### **Custom Model Integration**
```python
# The API automatically detects available models
# Check /status endpoint for model availability
```

### **Batch Processing with Python**
```python
import asyncio
import aiohttp

async def process_multiple_datasets(dataset_ids):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for dataset_id in dataset_ids:
            task = session.post(
                'http://localhost:8000/process-all',
                json={'dataset_id': dataset_id}
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
```

### **Integration with External Systems**
```python
# Webhook for processing completion
webhook_url = "https://your-system.com/webhook"

response = requests.post(
    'http://localhost:8000/process-all',
    json={
        'dataset_id': dataset_id,
        'config': {
            'webhook_url': webhook_url
        }
    }
)
```

## **Support & Contact**

For issues or questions:
1. Check the logs first
2. Verify input formats
3. Test with small datasets
4. Check system resources

The API is designed to be robust and provide detailed error messages for troubleshooting.
