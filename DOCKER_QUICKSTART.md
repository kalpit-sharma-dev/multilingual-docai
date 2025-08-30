# 🐳 PS-05 Challenge - Docker Quick Start Guide

## 🚀 **One-Command Deployment**

### **Linux/Mac:**
```bash
chmod +x build.sh
./build.sh
```

### **Windows:**
```cmd
build.bat
```

## 📋 **Prerequisites**

- ✅ Docker Desktop installed and running
- ✅ Docker Compose available
- ✅ At least 8GB RAM available
- ✅ 20GB free disk space

## 🔧 **Manual Setup (Alternative)**

### **1. Build the Image**
```bash
docker build -t ps05-backend:latest .
```

### **2. Create Directories**
```bash
mkdir -p datasets results logs models data/api_datasets data/api_results
mkdir -p nginx/ssl monitoring/grafana/dashboards monitoring/grafana/datasources
```

### **3. Generate SSL Certificates**
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/key.pem \
    -out nginx/ssl/cert.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

### **4. Start Services**
```bash
docker-compose up -d
```

## 🌐 **Service URLs**

| Service | URL | Description |
|---------|-----|-------------|
| **Backend API** | http://localhost:8000 | Main PS-05 API |
| **API Docs** | http://localhost:8000/docs | Swagger Documentation |
| **Nginx** | http://localhost:80 | Load Balancer |
| **Upload Service** | http://localhost:8080 | Large File Uploads |
| **Prometheus** | http://localhost:9090 | Metrics & Monitoring |
| **Grafana** | http://localhost:3000 | Dashboard (admin/admin) |

## 📊 **Monitor Your Deployment**

### **Check Service Status**
```bash
docker-compose ps
```

### **View Logs**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ps05-backend
```

### **Health Check**
```bash
curl http://localhost:8000/health
```

## 🎯 **Test Your Deployment**

### **1. Upload Dataset**
```bash
curl -X POST "http://localhost:8000/upload-dataset" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@test_image.jpg" \
  -F "dataset_name=test_dataset"
```

### **2. Process Dataset**
```bash
curl -X POST "http://localhost:8000/process-all" \
  -H "accept: application/json" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "dataset_id=YOUR_DATASET_ID&parallel_processing=true&gpu_acceleration=true&batch_size=50&optimization_level=speed"
```

### **3. Check Status**
```bash
curl http://localhost:8000/processing-stats
```

## 🔍 **Troubleshooting**

### **Service Won't Start**
```bash
# Check Docker logs
docker-compose logs ps05-backend

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

### **Port Already in Use**
```bash
# Check what's using the port
netstat -tulpn | grep :8000

# Kill the process or change ports in docker-compose.yml
```

### **Out of Memory**
```bash
# Check Docker memory usage
docker stats

# Increase Docker Desktop memory limit
# Docker Desktop → Settings → Resources → Memory
```

### **SSL Certificate Issues**
```bash
# Regenerate certificates
rm -rf nginx/ssl/*
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/key.pem \
    -out nginx/ssl/cert.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

## 🚀 **Production Deployment**

### **Use Production Compose**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### **Environment Variables**
```bash
export POSTGRES_PASSWORD="your_secure_password"
export GRAFANA_PASSWORD="your_grafana_password"
docker-compose -f docker-compose.prod.yml up -d
```

### **Scale Services**
```bash
# Scale backend to multiple instances
docker-compose up -d --scale ps05-backend=3
```

## 📁 **Directory Structure**

```
ps05-challenge/
├── backend/                 # Backend application code
├── scripts/                 # Utility scripts
├── datasets/                # Uploaded datasets
├── results/                 # Processing results
├── models/                  # ML models
├── logs/                    # Application logs
├── nginx/                   # Nginx configuration
│   ├── nginx.conf          # Main config
│   └── ssl/                # SSL certificates
├── monitoring/              # Monitoring configs
│   ├── prometheus.yml      # Prometheus config
│   └── grafana/            # Grafana dashboards
├── Dockerfile               # Main container
├── docker-compose.yml       # Development setup
├── docker-compose.prod.yml  # Production setup
├── build.sh                 # Linux/Mac build script
├── build.bat                # Windows build script
└── requirements.txt         # Python dependencies
```

## 🛠️ **Useful Commands**

### **Development**
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart specific service
docker-compose restart ps05-backend

# View logs
docker-compose logs -f ps05-backend
```

### **Production**
```bash
# Start production services
docker-compose -f docker-compose.prod.yml up -d

# Stop production services
docker-compose -f docker-compose.prod.yml down

# Backup data
docker-compose -f docker-compose.prod.yml run backup

# Update services
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

### **Maintenance**
```bash
# Clean up unused images
docker image prune -a

# Clean up unused volumes
docker volume prune

# Clean up everything
docker system prune -a

# Check disk usage
docker system df
```

## 🎉 **Success!**

Your PS-05 Challenge backend is now running in Docker! 

- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Monitoring**: http://localhost:3000

Ready to process 20GB datasets with GPU optimization! 🚀
