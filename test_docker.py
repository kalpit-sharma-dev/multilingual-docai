#!/usr/bin/env python3
"""
Docker Test Script for PS-05 Challenge
Tests the Docker setup and basic functionality
"""

import subprocess
import time
import requests
import json

def run_command(cmd):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_docker_build():
    """Test Docker build"""
    print("🔨 Testing Docker build...")
    success, stdout, stderr = run_command("docker build -t ps05-backend:test .")
    if success:
        print("✅ Docker build successful!")
        return True
    else:
        print("❌ Docker build failed:")
        print(stderr)
        return False

def test_docker_run():
    """Test running the container"""
    print("🚀 Testing Docker run...")
    
    # Stop any existing container
    run_command("docker stop ps05-backend-test 2>/dev/null || true")
    run_command("docker rm ps05-backend-test 2>/dev/null || true")
    
    # Start container
    success, stdout, stderr = run_command(
        "docker run -d --name ps05-backend-test -p 8001:8000 ps05-backend:test"
    )
    
    if success:
        print("✅ Container started successfully!")
        
        # Wait for container to be ready
        print("⏳ Waiting for container to be ready...")
        time.sleep(10)
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:8001/health", timeout=10)
            if response.status_code == 200:
                print("✅ Health check passed!")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    else:
        print("❌ Container start failed:")
        print(stderr)
        return False

def test_docker_compose():
    """Test Docker Compose"""
    print("🐳 Testing Docker Compose...")
    
    # Stop any existing services
    run_command("docker compose down 2>/dev/null || true")
    
    # Start services
    success, stdout, stderr = run_command("docker compose up -d")
    
    if success:
        print("✅ Docker Compose started successfully!")
        
        # Wait for services to be ready
        print("⏳ Waiting for services to be ready...")
        time.sleep(15)
        
        # Check service status
        success, stdout, stderr = run_command("docker compose ps")
        if success:
            print("📊 Service Status:")
            print(stdout)
        
        return True
    else:
        print("❌ Docker Compose failed:")
        print(stderr)
        return False

def cleanup():
    """Clean up test containers"""
    print("🧹 Cleaning up...")
    run_command("docker stop ps05-backend-test 2>/dev/null || true")
    run_command("docker rm ps05-backend-test 2>/dev/null || true")
    run_command("docker compose down 2>/dev/null || true")

def main():
    """Main test function"""
    print("🧪 PS-05 Challenge - Docker Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Docker build
        if not test_docker_build():
            return False
        
        # Test 2: Docker run
        if not test_docker_run():
            return False
        
        # Test 3: Docker Compose
        if not test_docker_compose():
            return False
        
        print("\n🎉 All tests passed! Docker setup is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    finally:
        cleanup()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
