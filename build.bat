@echo off
REM PS-05 Challenge - Docker Build and Deployment Script (Windows)
REM This script builds and deploys the complete PS-05 solution

echo 🚀 PS-05 Challenge - Docker Build and Deployment
echo ==================================================

REM Check if Docker is running
echo [INFO] Checking Docker installation...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)
echo [SUCCESS] Docker is running

REM Check if Docker Compose is available
echo [INFO] Checking Docker Compose...
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose is not available. Please install Docker Compose and try again.
    pause
    exit /b 1
)
echo [SUCCESS] Docker Compose is available

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "datasets" mkdir datasets
if not exist "results" mkdir results
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "data\api_datasets" mkdir data\api_datasets
if not exist "data\api_results" mkdir data\api_results
if not exist "nginx\ssl" mkdir nginx\ssl
if not exist "monitoring\grafana\dashboards" mkdir monitoring\grafana\dashboards
if not exist "monitoring\grafana\datasources" mkdir monitoring\grafana\datasources
echo [SUCCESS] Directories created

REM Generate self-signed SSL certificates for development
echo [INFO] Generating self-signed SSL certificates...
if not exist "nginx\ssl\cert.pem" (
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout nginx\ssl\key.pem -out nginx\ssl\cert.pem -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    echo [SUCCESS] SSL certificates generated
) else (
    echo [INFO] SSL certificates already exist
)

REM Build Docker images
echo [INFO] Building Docker images...
docker build -t ps05-backend:latest .
if %errorlevel% neq 0 (
    echo [ERROR] Docker build failed
    pause
    exit /b 1
)
echo [SUCCESS] Docker images built successfully

REM Start services
echo [INFO] Starting PS-05 services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services
    pause
    exit /b 1
)
echo [SUCCESS] Services started successfully

REM Wait for services to be ready
echo [INFO] Waiting for services to be ready...
echo [INFO] Waiting for backend service...
:wait_loop
timeout /t 5 /nobreak >nul
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo -n .
    goto wait_loop
)
echo.
echo [SUCCESS] Backend service is ready

echo [INFO] Waiting for other services...
timeout /t 10 /nobreak >nul
echo [SUCCESS] All services are ready

REM Show service status
echo [INFO] Service Status:
docker-compose ps

echo.
echo [INFO] Service URLs:
echo   Backend API: http://localhost:8000
echo   Nginx Load Balancer: http://localhost:80
echo   Upload Service: http://localhost:8080
echo   Prometheus Monitoring: http://localhost:9090
echo   Grafana Dashboard: http://localhost:3000
echo   API Documentation: http://localhost:8000/docs

echo.
echo [SUCCESS] 🎉 PS-05 Challenge deployment completed successfully!
echo.
echo [INFO] Next steps:
echo   1. Upload your dataset: curl -X POST http://localhost:8000/upload-dataset
echo   2. Start processing: curl -X POST http://localhost:8000/process-all
echo   3. Monitor progress: curl http://localhost:8000/processing-stats
echo   4. View results: http://localhost:8000/results
echo.
echo [INFO] To view logs: docker-compose logs -f ps05-backend
echo [INFO] To stop services: docker-compose down

pause
