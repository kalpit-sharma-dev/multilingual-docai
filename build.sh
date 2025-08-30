#!/bin/bash

# PS-05 Challenge - Docker Build and Deployment Script
# This script builds and deploys the complete PS-05 solution

set -e

echo "🚀 PS-05 Challenge - Docker Build and Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker installation..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    if ! docker-compose --version > /dev/null 2>&1; then
        print_error "Docker Compose is not available. Please install Docker Compose and try again."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p datasets results logs models data/api_datasets data/api_results
    mkdir -p nginx/ssl monitoring/grafana/dashboards monitoring/grafana/datasources
    
    print_success "Directories created"
}

# Generate self-signed SSL certificates for development
generate_ssl_certs() {
    print_status "Generating self-signed SSL certificates..."
    
    if [ ! -f "nginx/ssl/cert.pem" ] || [ ! -f "nginx/ssl/key.pem" ]; then
        mkdir -p nginx/ssl
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/key.pem \
            -out nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        print_success "SSL certificates generated"
    else
        print_status "SSL certificates already exist"
    fi
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build the main backend image
    docker build -t ps05-backend:latest .
    
    print_success "Docker images built successfully"
}

# Start services
start_services() {
    print_status "Starting PS-05 services..."
    
    # Start all services
    docker-compose up -d
    
    print_success "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for backend
    print_status "Waiting for backend service..."
    until curl -f http://localhost:8000/health > /dev/null 2>&1; do
        echo -n "."
        sleep 5
    done
    echo ""
    print_success "Backend service is ready"
    
    # Wait for other services
    print_status "Waiting for other services..."
    sleep 10
    
    print_success "All services are ready"
}

# Show service status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    echo ""
    print_status "Service URLs:"
    echo -e "  ${GREEN}Backend API:${NC} http://localhost:8000"
    echo -e "  ${GREEN}Nginx Load Balancer:${NC} http://localhost:80"
    echo -e "  ${GREEN}Upload Service:${NC} http://localhost:8080"
    echo -e "  ${GREEN}Prometheus Monitoring:${NC} http://localhost:9090"
    echo -e "  ${GREEN}Grafana Dashboard:${NC} http://localhost:3000"
    echo -e "  ${GREEN}API Documentation:${NC} http://localhost:8000/docs"
}

# Show logs
show_logs() {
    print_status "Recent logs from backend service:"
    docker-compose logs --tail=20 ps05-backend
}

# Main build function
main() {
    echo ""
    print_status "Starting PS-05 Docker build and deployment..."
    
    # Pre-flight checks
    check_docker
    check_docker_compose
    
    # Setup
    create_directories
    generate_ssl_certs
    
    # Build and deploy
    build_images
    start_services
    wait_for_services
    
    # Show results
    show_status
    
    echo ""
    print_success "🎉 PS-05 Challenge deployment completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "  1. Upload your dataset: curl -X POST http://localhost:8000/upload-dataset"
    echo "  2. Start processing: curl -X POST http://localhost:8000/process-all"
    echo "  3. Monitor progress: curl http://localhost:8000/processing-stats"
    echo "  4. View results: http://localhost:8000/results"
    echo ""
    print_status "To view logs: docker-compose logs -f ps05-backend"
    print_status "To stop services: docker-compose down"
}

# Run main function
main "$@"
