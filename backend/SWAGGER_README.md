# 🚀 PS-05 Backend API - Swagger Documentation

## 📖 Overview

This directory contains the comprehensive Swagger/OpenAPI 3.0 documentation for the PS-05 Document Understanding Backend API. The documentation provides detailed information about all available endpoints, request/response models, and usage examples.

## 📁 Files

- **`swagger_documentation.yaml`** - Complete OpenAPI 3.0 specification
- **`SWAGGER_README.md`** - This documentation file

## 🌐 Accessing the Documentation

### **1. Interactive Swagger UI**
When the backend is running, access the interactive documentation at:
```
http://localhost:8000/docs
```

### **2. ReDoc Alternative**
For an alternative documentation view:
```
http://localhost:8000/redoc
```

### **3. Raw YAML**
View the raw OpenAPI specification:
```
http://localhost:8000/openapi.json
```

## 🔧 API Endpoints Overview

### **System & Health**
- **`GET /`** - API information and capabilities
- **`GET /status`** - System status and model availability
- **`GET /health`** - Health check for monitoring

### **Dataset Management**
- **`POST /upload-dataset`** - Upload datasets (with/without annotations)
- **`GET /datasets`** - List all uploaded datasets
- **`DELETE /datasets/{dataset_id}`** - Delete datasets and results

### **Core Processing**
- **`POST /process-stage`** - Process specific stage (1, 2, or 3)
- **`POST /process-all`** - Process all stages in background
- **`GET /results/{dataset_id}`** - Get processing results

### **Evaluation & Predictions**
- **`POST /evaluate`** - Evaluate with annotations (mAP calculation)
- **`GET /predictions/{dataset_id}`** - Get predictions (no annotations)

### **Data Cleaning**
- **`POST /clean-dataset`** - Comprehensive data cleaning
- **`GET /cleaning-capabilities`** - Available cleaning features
- **`GET /cleaning-status/{dataset_id}`** - Cleaning job status

### **EDA Analysis**
- **`POST /run-eda`** - Exploratory Data Analysis
- **`GET /eda-results/{dataset_id}`** - EDA results retrieval

## 🎯 Key Features Documented

### **1. Complete 3-Stage Pipeline**
- **Stage 1**: Layout Detection (YOLOv8, LayoutLMv3, Mask R-CNN)
- **Stage 2**: Text Extraction + Language ID (EasyOCR, Tesseract, fastText)
- **Stage 3**: Content Understanding + NLG (Table Transformer, BLIP, OFA)

### **2. Evaluation Modes**
- **With Annotations**: Full mAP evaluation and metrics
- **Without Annotations**: Prediction-only mode for evaluator testing

### **3. Data Cleaning Services**
- **Image Cleaning**: 10 comprehensive cleaning tasks
- **Document Cleaning**: 11 document processing tasks
- **Unified Workflow**: Integrated cleaning pipeline

### **4. EDA Integration**
- **File Analysis**: Format, size, and type analysis
- **Image Properties**: Dimensions, rotation, quality assessment
- **Visualizations**: Charts, plots, and statistical insights

### **5. Large Dataset Support**
- **20GB+ Support**: Optimized for large-scale processing
- **Background Jobs**: Non-blocking operations
- **Memory Optimization**: Efficient resource usage

## 📊 Data Models

### **Core Models**
- **`ProcessingRequest`** - Stage processing requests
- **`DatasetUploadResponse`** - Upload confirmation
- **`StageResult`** - Stage processing results
- **`EvaluationResult`** - Complete evaluation metrics

### **Data Structures**
- **`BoundingBox`** - Coordinate system for layout elements
- **`LayoutElement`** - Detected document components
- **`NoAnnotationResponse`** - Prediction-only results

### **System Models**
- **`SystemStatus`** - API health and capabilities
- **`ErrorResponse`** - Standardized error handling

## 🚀 Getting Started

### **1. Start the Backend**
```bash
# From project root
python ps05.py backend --start

# Or directly
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### **2. Access Documentation**
```bash
# Open in browser
http://localhost:8000/docs
```

### **3. Test Endpoints**
Use the interactive Swagger UI to:
- Test API endpoints directly
- View request/response schemas
- Execute sample requests
- Download OpenAPI specification

## 🔍 Using the Swagger UI

### **1. Endpoint Exploration**
- **Expand** endpoint sections to see available operations
- **Click** on endpoints to view detailed information
- **Try it out** button to execute requests directly

### **2. Request Building**
- **Parameters**: Fill in required path/query parameters
- **Request Body**: Use the provided schema examples
- **File Uploads**: Use the file upload interface for multipart requests

### **3. Response Analysis**
- **Response Codes**: View all possible HTTP status codes
- **Response Schema**: Understand the data structure
- **Examples**: See sample responses for each endpoint

## 📋 API Testing Examples

### **1. Upload Dataset**
```bash
curl -X POST "http://localhost:8000/upload-dataset" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sample_image.jpg" \
  -F "dataset_name=test_dataset"
```

### **2. Process All Stages**
```bash
curl -X POST "http://localhost:8000/process-all" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "your-dataset-id"}'
```

### **3. Get Predictions**
```bash
curl -X GET "http://localhost:8000/predictions/your-dataset-id" \
  -H "accept: application/json"
```

## 🛠️ Development & Customization

### **1. Modify Documentation**
Edit `swagger_documentation.yaml` to:
- Add new endpoints
- Update request/response models
- Modify examples and descriptions
- Add authentication schemes

### **2. Regenerate OpenAPI**
The FastAPI app automatically generates OpenAPI from the code:
```python
# In main.py
app = FastAPI(
    title="PS-05 Document Understanding API",
    description="Complete 3-stage document understanding pipeline",
    version="1.0.0"
)
```

### **3. Custom Tags & Grouping**
Organize endpoints using tags:
```python
@app.post("/upload-dataset", tags=["Dataset Management"])
async def upload_dataset():
    # Implementation
```

## 🔐 Security & Authentication

### **Current Status**
- **No Authentication**: Currently open access
- **Production Ready**: Framework supports JWT, API keys, OAuth2

### **Future Implementation**
```yaml
components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

## 📈 Monitoring & Health

### **Health Check**
```bash
curl http://localhost:8000/health
# Response: {"status": "healthy", "version": "1.0.0"}
```

### **System Status**
```bash
curl http://localhost:8000/status
# Response: Complete system information
```

## 🐳 Docker Integration

### **Swagger in Docker**
```dockerfile
# The Swagger UI is automatically available
# when running the FastAPI app in Docker
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Access from Container**
```bash
# From host machine
http://localhost:8000/docs

# From container network
http://container-ip:8000/docs
```

## 🔧 Troubleshooting

### **Common Issues**

#### **1. Documentation Not Loading**
- Check if backend is running: `curl http://localhost:8000/health`
- Verify port 8000 is accessible
- Check firewall settings

#### **2. Schema Validation Errors**
- Ensure YAML syntax is correct
- Validate against OpenAPI 3.0 specification
- Check for missing required fields

#### **3. Endpoint Not Found**
- Verify endpoint path in `main.py`
- Check if endpoint is properly decorated
- Ensure FastAPI app is restarted after changes

### **Validation Tools**
```bash
# Validate OpenAPI specification
pip install openapi-spec-validator
openapi-spec-validator backend/swagger_documentation.yaml

# Online validation
# https://editor.swagger.io/
```

## 📚 Additional Resources

### **OpenAPI Documentation**
- [OpenAPI 3.0 Specification](https://swagger.io/specification/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Swagger UI Documentation](https://swagger.io/tools/swagger-ui/)

### **PS-05 Project Documentation**
- **`README_MASTER.md`** - Complete project overview
- **`PROJECT_STRUCTURE.md`** - High-level architecture
- **`DIRECTORY_STRUCTURE.md`** - Detailed file layout
- **`NAVIGATION_INDEX.md`** - Quick reference guide

## 🎉 Conclusion

The Swagger documentation provides a comprehensive, interactive interface for understanding and testing the PS-05 Backend API. It serves as both documentation and a testing tool, making it easy for developers and evaluators to work with the system.

### **Key Benefits**
- **Interactive Testing**: Test endpoints directly from the browser
- **Complete Coverage**: All endpoints, models, and examples documented
- **Professional Quality**: Enterprise-grade API documentation
- **Easy Maintenance**: Auto-generated from FastAPI code
- **Standards Compliant**: OpenAPI 3.0 specification

### **Next Steps**
1. **Start the backend**: `python ps05.py backend --start`
2. **Access documentation**: `http://localhost:8000/docs`
3. **Explore endpoints**: Use the interactive interface
4. **Test functionality**: Execute sample requests
5. **Customize as needed**: Modify for your specific requirements

---

**🚀 Ready to explore the PS-05 Document Understanding API!**
