#!/usr/bin/env python3
"""
Test runner script for the cyberattack detection system.
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Command failed: {command}")
            print(f"Error: {result.stderr}")
            return False
        
        print(f"Command succeeded: {command}")
        if result.stdout:
            print(result.stdout)
        
        return True
        
    except Exception as e:
        print(f"Error running command {command}: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Cyberattack Detection System Tests")
    print("=" * 50)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run different test suites
    test_suites = [
        {
            "name": "Model Tests",
            "command": "python -m pytest tests/test_models.py -v"
        },
        {
            "name": "API Tests", 
            "command": "python -m pytest tests/test_api.py -v"
        },
        {
            "name": "Feature Tests",
            "command": "python -m pytest tests/test_features.py -v"
        },
        {
            "name": "All Tests",
            "command": "python -m pytest tests/ -v --tb=short"
        }
    ]
    
    results = []
    
    for suite in test_suites:
        print(f"\n🔍 Running {suite['name']}...")
        print("-" * 30)
        
        success = run_command(suite['command'])
        results.append((suite['name'], success))
        
        if not success:
            print(f"❌ {suite['name']} failed")
        else:
            print(f"✅ {suite['name']} passed")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} test suites")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
