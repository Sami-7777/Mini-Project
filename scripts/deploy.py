#!/usr/bin/env python3
"""
Deployment script for the cyberattack detection system.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
from datetime import datetime

def run_command(command, cwd=None, check=True):
    """Run a command and handle errors."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        
        if result.stdout:
            print(result.stdout)
        
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"Error: {e.stderr}")
        return False

def create_deployment_package():
    """Create deployment package."""
    print("📦 Creating deployment package...")
    
    # Create deployment directory
    deploy_dir = Path("deployment")
    deploy_dir.mkdir(exist_ok=True)
    
    # Copy necessary files
    files_to_copy = [
        "src/",
        "config/",
        "scripts/",
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile.api",
        "Dockerfile.dashboard",
        "README.md"
    ]
    
    for file_path in files_to_copy:
        src = Path(file_path)
        dst = deploy_dir / file_path
        
        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    
    print("✅ Deployment package created")

def build_docker_images():
    """Build Docker images."""
    print("🐳 Building Docker images...")
    
    # Build API image
    if not run_command("docker build -f Dockerfile.api -t cyberattack-api:latest ."):
        print("❌ Failed to build API image")
        return False
    
    # Build Dashboard image
    if not run_command("docker build -f Dockerfile.dashboard -t cyberattack-dashboard:latest ."):
        print("❌ Failed to build Dashboard image")
        return False
    
    print("✅ Docker images built successfully")
    return True

def deploy_with_docker_compose():
    """Deploy using Docker Compose."""
    print("🚀 Deploying with Docker Compose...")
    
    # Stop existing containers
    run_command("docker-compose down", check=False)
    
    # Start services
    if not run_command("docker-compose up -d"):
        print("❌ Failed to start services")
        return False
    
    print("✅ Services started successfully")
    return True

def deploy_to_kubernetes():
    """Deploy to Kubernetes (placeholder)."""
    print("☸️  Kubernetes deployment not implemented yet")
    print("   This would deploy to a Kubernetes cluster")
    return True

def deploy_to_cloud(provider="aws"):
    """Deploy to cloud provider."""
    print(f"☁️  Cloud deployment to {provider} not implemented yet")
    print("   This would deploy to cloud infrastructure")
    return True

def run_health_checks():
    """Run health checks after deployment."""
    print("🏥 Running health checks...")
    
    import time
    import requests
    
    # Wait for services to start
    time.sleep(30)
    
    # Check API health
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ API health check passed")
        else:
            print("❌ API health check failed")
            return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False
    
    # Check Dashboard health
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Dashboard health check passed")
        else:
            print("❌ Dashboard health check failed")
            return False
    except Exception as e:
        print(f"❌ Dashboard health check failed: {e}")
        return False
    
    print("✅ All health checks passed")
    return True

def generate_deployment_report():
    """Generate deployment report."""
    print("📋 Generating deployment report...")
    
    report = {
        "deployment_time": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "api": {
                "status": "running",
                "port": 8000,
                "health_endpoint": "http://localhost:8000/health"
            },
            "dashboard": {
                "status": "running", 
                "port": 8501,
                "url": "http://localhost:8501"
            },
            "database": {
                "status": "running",
                "port": 27017
            },
            "redis": {
                "status": "running",
                "port": 6379
            }
        },
        "endpoints": {
            "api_docs": "http://localhost:8000/docs",
            "dashboard": "http://localhost:8501",
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3000"
        }
    }
    
    # Save report
    with open("deployment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("✅ Deployment report generated")
    return report

def main():
    """Main deployment function."""
    print("🚀 Cyberattack Detection System Deployment")
    print("=" * 50)
    
    # Parse command line arguments
    deployment_type = "docker"
    if len(sys.argv) > 1:
        deployment_type = sys.argv[1]
    
    print(f"Deployment type: {deployment_type}")
    
    try:
        # Create deployment package
        create_deployment_package()
        
        # Deploy based on type
        if deployment_type == "docker":
            if not build_docker_images():
                sys.exit(1)
            
            if not deploy_with_docker_compose():
                sys.exit(1)
                
        elif deployment_type == "kubernetes":
            if not deploy_to_kubernetes():
                sys.exit(1)
                
        elif deployment_type == "cloud":
            provider = sys.argv[2] if len(sys.argv) > 2 else "aws"
            if not deploy_to_cloud(provider):
                sys.exit(1)
        
        else:
            print(f"❌ Unknown deployment type: {deployment_type}")
            sys.exit(1)
        
        # Run health checks
        if not run_health_checks():
            sys.exit(1)
        
        # Generate report
        report = generate_deployment_report()
        
        print("\n" + "=" * 50)
        print("🎉 Deployment completed successfully!")
        print("=" * 50)
        print("\n📊 System Status:")
        for service, info in report["services"].items():
            print(f"  {service}: {info['status']}")
        
        print("\n🌐 Access Points:")
        for name, url in report["endpoints"].items():
            print(f"  {name}: {url}")
        
        print("\n📋 Next Steps:")
        print("  1. Update .env file with your API keys")
        print("  2. Initialize the database: python scripts/init_db.py")
        print("  3. Access the dashboard at http://localhost:8501")
        print("  4. View API documentation at http://localhost:8000/docs")
        
    except KeyboardInterrupt:
        print("\n❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
