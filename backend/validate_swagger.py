#!/usr/bin/env python3
"""
Swagger Documentation Validation Script

This script validates the OpenAPI 3.0 specification file
and checks for common issues.
"""

import yaml
import json
import sys
from pathlib import Path

def validate_yaml_syntax(file_path):
    """Validate YAML syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        print("✅ YAML syntax is valid")
        return True
    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error: {e}")
        return False

def validate_openapi_structure(file_path):
    """Validate OpenAPI 3.0 structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            spec = yaml.safe_load(f)
        
        # Check required OpenAPI fields
        required_fields = ['openapi', 'info', 'paths']
        for field in required_fields:
            if field not in spec:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check OpenAPI version
        if not spec['openapi'].startswith('3.'):
            print(f"❌ Invalid OpenAPI version: {spec['openapi']}")
            return False
        
        # Check info section
        info_required = ['title', 'version']
        for field in info_required:
            if field not in spec['info']:
                print(f"❌ Missing info field: {field}")
                return False
        
        # Check paths
        if not spec['paths']:
            print("❌ No paths defined")
            return False
        
        print("✅ OpenAPI 3.0 structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ OpenAPI validation error: {e}")
        return False

def count_endpoints(file_path):
    """Count the number of endpoints."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            spec = yaml.safe_load(f)
        
        endpoint_count = 0
        for path, methods in spec['paths'].items():
            for method in methods:
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    endpoint_count += 1
        
        print(f"📊 Found {endpoint_count} endpoints")
        return endpoint_count
        
    except Exception as e:
        print(f"❌ Error counting endpoints: {e}")
        return 0

def check_schemas(file_path):
    """Check schema definitions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            spec = yaml.safe_load(f)
        
        if 'components' in spec and 'schemas' in spec['components']:
            schema_count = len(spec['components']['schemas'])
            print(f"📊 Found {schema_count} schema definitions")
            return schema_count
        else:
            print("⚠️ No schemas defined")
            return 0
            
    except Exception as e:
        print(f"❌ Error checking schemas: {e}")
        return 0

def generate_summary(file_path):
    """Generate a summary of the API."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            spec = yaml.safe_load(f)
        
        print("\n📋 API Summary:")
        print(f"   Title: {spec['info']['title']}")
        print(f"   Version: {spec['info']['version']}")
        print(f"   Description: {spec['info']['description'][:100]}...")
        
        # Count endpoints by tag
        tag_counts = {}
        for path, methods in spec['paths'].items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    if 'tags' in details:
                        for tag in details['tags']:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        if tag_counts:
            print("\n   Endpoints by Tag:")
            for tag, count in tag_counts.items():
                print(f"     {tag}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return False

def main():
    """Main validation function."""
    swagger_file = Path(__file__).parent / "swagger_documentation.yaml"
    
    if not swagger_file.exists():
        print(f"❌ Swagger file not found: {swagger_file}")
        sys.exit(1)
    
    print("🔍 Validating PS-05 Swagger Documentation")
    print("=" * 50)
    
    # Run validations
    yaml_valid = validate_yaml_syntax(swagger_file)
    openapi_valid = validate_openapi_structure(swagger_file)
    endpoint_count = count_endpoints(swagger_file)
    schema_count = check_schemas(swagger_file)
    summary_generated = generate_summary(swagger_file)
    
    print("\n" + "=" * 50)
    
    # Overall validation result
    if all([yaml_valid, openapi_valid, endpoint_count > 0, schema_count > 0]):
        print("🎉 Swagger documentation validation PASSED!")
        print(f"📊 Total endpoints: {endpoint_count}")
        print(f"📊 Total schemas: {schema_count}")
        print("\n✅ Ready to use with FastAPI!")
        return True
    else:
        print("❌ Swagger documentation validation FAILED!")
        print("Please fix the issues above before using.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
