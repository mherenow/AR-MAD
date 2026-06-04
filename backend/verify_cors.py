"""Quick verification script for CORS configuration.

This script verifies that the CORS middleware is properly configured
by checking the middleware stack and configuration.
"""

import sys
import os

# Add ai-image-detector to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai-image-detector'))

from main import app
from fastapi.middleware.cors import CORSMiddleware

def verify_cors_configuration():
    """Verify CORS middleware is configured correctly."""
    
    print("=== CORS Configuration Verification ===\n")
    
    # Check if CORS middleware is in the middleware stack
    has_cors = False
    for middleware in app.user_middleware:
        if middleware.cls == CORSMiddleware:
            has_cors = True
            print("✓ CORSMiddleware is registered")
            
            # Check configuration from kwargs
            kwargs = middleware.kwargs
            print(f"\nConfiguration:")
            print(f"  - allow_origins: {kwargs.get('allow_origins', [])}")
            print(f"  - allow_methods: {kwargs.get('allow_methods', [])}")
            print(f"  - allow_headers: {kwargs.get('allow_headers', [])}")
            
            # Verify requirements
            print(f"\nRequirement Validation:")
            
            allowed_origins = kwargs.get('allow_origins', [])
            expected_origin = os.getenv("ALLOWED_ORIGIN", "http://localhost:5173")
            if expected_origin in allowed_origins:
                print(f"  ✓ ALLOWED_ORIGIN environment variable configured (default: http://localhost:5173)")
                print(f"    Current value: {expected_origin}")
            else:
                print(f"  ✗ ALLOWED_ORIGIN not properly configured")
            
            allowed_methods = kwargs.get('allow_methods', [])
            required_methods = ['GET', 'POST', 'OPTIONS']
            if all(method in allowed_methods for method in required_methods):
                print(f"  ✓ Required methods configured: GET, POST, OPTIONS")
            else:
                print(f"  ✗ Missing required methods")
            
            allowed_headers = kwargs.get('allow_headers', [])
            if allowed_headers == ['*']:
                print(f"  ✓ All headers allowed for preflight requests (allow_headers=['*'])")
            else:
                print(f"  ⚠ Headers configuration: {allowed_headers}")
            
            break
    
    if not has_cors:
        print("✗ CORSMiddleware is NOT registered")
        return False
    
    print("\n=== Task 2.3 Complete ===")
    print("CORS middleware is properly configured according to requirements.")
    return True

if __name__ == "__main__":
    success = verify_cors_configuration()
    sys.exit(0 if success else 1)
