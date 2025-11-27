#!/usr/bin/env python3
"""
Setup script for the cyberattack detection system.
"""
import os
import sys
import subprocess
import asyncio
from pathlib import Path
import shutil

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import settings
from core.logger import logger


def run_command(command: str, cwd: str = None) -> bool:
    """Run a shell command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error("Command failed", 
                        command=command, 
                        returncode=result.returncode,
                        stderr=result.stderr)
            return False
        
        logger.info("Command completed", command=command)
        return True
        
    except Exception as e:
        logger.error("Error running command", command=command, error=str(e))
        return False


def create_directories():
    """Create necessary directories."""
    try:
        directories = [
            "data/models",
            "data/logs",
            "logs",
            "config",
            "monitoring/prometheus",
            "monitoring/grafana/dashboards",
            "monitoring/grafana/datasources",
            "nginx/ssl"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info("Created directory", directory=directory)
        
    except Exception as e:
        logger.error("Error creating directories", error=str(e))
        raise


def install_dependencies():
    """Install Python dependencies."""
    try:
        logger.info("Installing Python dependencies")
        
        # Install requirements
        if not run_command("pip install -r requirements.txt"):
            raise Exception("Failed to install requirements")
        
        # Install additional dependencies
        additional_packages = [
            "streamlit-folium",
            "prometheus-client",
            "psutil"
        ]
        
        for package in additional_packages:
            if not run_command(f"pip install {package}"):
                logger.warning("Failed to install package", package=package)
        
        logger.info("Dependencies installed successfully")
        
    except Exception as e:
        logger.error("Error installing dependencies", error=str(e))
        raise


def setup_environment():
    """Setup environment variables."""
    try:
        logger.info("Setting up environment")
        
        # Create .env file if it doesn't exist
        env_file = Path(".env")
        if not env_file.exists():
            # Copy from example
            example_file = Path("env.example")
            if example_file.exists():
                shutil.copy(example_file, env_file)
                logger.info("Created .env file from example")
            else:
                # Create basic .env file
                env_content = """# Database Configuration
MONGODB_URL=mongodb://localhost:27017/cyberattack_db
REDIS_URL=redis://localhost:6379

# API Keys (add your keys here)
VIRUSTOTAL_API_KEY=your_virustotal_api_key_here
GOOGLE_SAFE_BROWSING_API_KEY=your_google_safe_browsing_key_here
ABUSEIPDB_API_KEY=your_abuseipdb_api_key_here

# Security Configuration
SECRET_KEY=your-secret-key-change-in-production
ENCRYPTION_KEY=your-encryption-key-change-in-production

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/cyberattack_detection.log
"""
                
                with open(env_file, 'w') as f:
                    f.write(env_content)
                
                logger.info("Created basic .env file")
        
        logger.info("Environment setup completed")
        
    except Exception as e:
        logger.error("Error setting up environment", error=str(e))
        raise


def setup_database():
    """Setup database."""
    try:
        logger.info("Setting up database")
        
        # Check if MongoDB is running
        if not run_command("mongosh --eval 'db.runCommand(\"ping\")'"):
            logger.warning("MongoDB is not running. Please start MongoDB first.")
            return False
        
        # Check if Redis is running
        if not run_command("redis-cli ping"):
            logger.warning("Redis is not running. Please start Redis first.")
            return False
        
        # Initialize database
        if not run_command("python scripts/init_db.py"):
            logger.warning("Failed to initialize database")
            return False
        
        logger.info("Database setup completed")
        return True
        
    except Exception as e:
        logger.error("Error setting up database", error=str(e))
        return False


def setup_monitoring():
    """Setup monitoring configuration."""
    try:
        logger.info("Setting up monitoring")
        
        # Create Prometheus configuration
        prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'cyberattack-detection'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
"""
        
        prometheus_file = Path("monitoring/prometheus.yml")
        with open(prometheus_file, 'w') as f:
            f.write(prometheus_config)
        
        logger.info("Created Prometheus configuration")
        
        # Create Grafana datasource configuration
        grafana_datasource = """
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
"""
        
        datasource_file = Path("monitoring/grafana/datasources/prometheus.yml")
        with open(datasource_file, 'w') as f:
            f.write(grafana_datasource)
        
        logger.info("Created Grafana datasource configuration")
        
        logger.info("Monitoring setup completed")
        
    except Exception as e:
        logger.error("Error setting up monitoring", error=str(e))
        raise


def setup_nginx():
    """Setup Nginx configuration."""
    try:
        logger.info("Setting up Nginx")
        
        # Create Nginx configuration
        nginx_config = """
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }
    
    upstream dashboard {
        server dashboard:8501;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location /api/ {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location / {
            proxy_pass http://dashboard;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
"""
        
        nginx_file = Path("nginx/nginx.conf")
        with open(nginx_file, 'w') as f:
            f.write(nginx_config)
        
        logger.info("Created Nginx configuration")
        
    except Exception as e:
        logger.error("Error setting up Nginx", error=str(e))
        raise


def main():
    """Main setup function."""
    try:
        logger.info("Starting system setup")
        
        # Create directories
        create_directories()
        
        # Install dependencies
        install_dependencies()
        
        # Setup environment
        setup_environment()
        
        # Setup monitoring
        setup_monitoring()
        
        # Setup Nginx
        setup_nginx()
        
        # Setup database (optional)
        setup_database()
        
        logger.info("System setup completed successfully")
        logger.info("Next steps:")
        logger.info("1. Update .env file with your API keys")
        logger.info("2. Start the system with: python scripts/start_system.py")
        logger.info("3. Or use Docker: docker-compose up -d")
        
    except Exception as e:
        logger.error("Setup failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
