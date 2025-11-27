# 🚀 Deployment Guide

This guide covers various deployment options for the Cyberattack Detection System.

## 📋 Prerequisites

- Python 3.9+
- Docker and Docker Compose
- MongoDB (or use Docker)
- Redis (or use Docker)
- API keys for threat intelligence services

## 🐳 Docker Deployment (Recommended)

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd cyberattack-detection-system

# Run setup
python scripts/setup.py

# Deploy with Docker
python scripts/deploy.py docker

# Or manually with Docker Compose
docker-compose up -d
```

### Services

The Docker deployment includes:

- **API Server**: FastAPI backend on port 8000
- **Dashboard**: Streamlit frontend on port 8501
- **MongoDB**: Database on port 27017
- **Redis**: Cache on port 6379
- **Nginx**: Reverse proxy on port 80
- **Prometheus**: Metrics on port 9090
- **Grafana**: Monitoring on port 3000

### Configuration

1. **Environment Variables**: Update `.env` file with your API keys
2. **Database Initialization**: Run `python scripts/init_db.py`
3. **Health Checks**: Verify all services are running

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.x

### Deploy

```bash
# Create namespace
kubectl create namespace cyberattack-detection

# Deploy with Helm (when charts are available)
helm install cyberattack-detection ./helm-charts/cyberattack-detection \
  --namespace cyberattack-detection

# Or deploy with kubectl
kubectl apply -f k8s/
```

### Scaling

```bash
# Scale API replicas
kubectl scale deployment cyberattack-api --replicas=3

# Scale dashboard replicas
kubectl scale deployment cyberattack-dashboard --replicas=2
```

## ☁️ Cloud Deployment

### AWS Deployment

```bash
# Deploy to AWS ECS
python scripts/deploy.py cloud aws

# Or use AWS CDK
cdk deploy CyberattackDetectionStack
```

### Azure Deployment

```bash
# Deploy to Azure Container Instances
python scripts/deploy.py cloud azure

# Or use Azure CLI
az container create --resource-group cyberattack-rg --name cyberattack-system
```

### GCP Deployment

```bash
# Deploy to Google Cloud Run
python scripts/deploy.py cloud gcp

# Or use gcloud
gcloud run deploy cyberattack-detection --source .
```

## 🔧 Manual Deployment

### System Requirements

- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 50GB+
- **Network**: 100Mbps+

### Installation Steps

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Database**

   ```bash
   # Start MongoDB
   sudo systemctl start mongod

   # Start Redis
   sudo systemctl start redis

   # Initialize database
   python scripts/init_db.py
   ```

3. **Configure Environment**

   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Start Services**

   ```bash
   # Start API server
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000

   # Start dashboard (in another terminal)
   streamlit run src/dashboard/main.py --server.port 8501
   ```

## 🔍 Health Checks

### Automated Health Checks

```bash
# Run health checks
python scripts/health_check.py

# Check specific service
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

### Manual Verification

1. **API Health**: http://localhost:8000/health
2. **Dashboard**: http://localhost:8501
3. **API Docs**: http://localhost:8000/docs
4. **Prometheus**: http://localhost:9090
5. **Grafana**: http://localhost:3000

## 📊 Monitoring

### Prometheus Metrics

- Request counts and latencies
- Model performance metrics
- System resource usage
- Threat detection statistics

### Grafana Dashboards

- System overview
- Performance metrics
- Threat analysis
- Alert management

### Logging

- Structured JSON logs
- Log rotation and retention
- Centralized log aggregation
- Error tracking and alerting

## 🔒 Security Configuration

### SSL/TLS Setup

```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Update nginx configuration
# Uncomment SSL server block in nginx.conf
```

### Firewall Configuration

```bash
# Allow necessary ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8000/tcp
sudo ufw allow 8501/tcp
```

### API Security

- API key authentication
- Rate limiting
- CORS configuration
- Input validation
- SQL injection prevention

## 🔄 Updates and Maintenance

### Rolling Updates

```bash
# Update with zero downtime
docker-compose up -d --no-deps api
docker-compose up -d --no-deps dashboard
```

### Backup Strategy

```bash
# Backup MongoDB
mongodump --db cyberattack_db --out backup/

# Backup Redis
redis-cli --rdb backup.rdb

# Backup models
tar -czf models_backup.tar.gz data/models/
```

### Monitoring Maintenance

- Regular log rotation
- Database cleanup
- Model retraining
- Security updates
- Performance optimization

## 🚨 Troubleshooting

### Common Issues

1. **Port Conflicts**

   ```bash
   # Check port usage
   netstat -tulpn | grep :8000

   # Kill process using port
   sudo kill -9 <PID>
   ```

2. **Database Connection Issues**

   ```bash
   # Check MongoDB status
   sudo systemctl status mongod

   # Check Redis status
   sudo systemctl status redis
   ```

3. **Memory Issues**

   ```bash
   # Check memory usage
   free -h

   # Increase swap if needed
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Log Analysis

```bash
# View API logs
docker-compose logs api

# View dashboard logs
docker-compose logs dashboard

# View system logs
journalctl -u cyberattack-detection
```

## 📞 Support

For deployment issues:

1. Check the logs for error messages
2. Verify all prerequisites are met
3. Ensure proper network connectivity
4. Review security configurations
5. Contact support with detailed error information

## 🔗 Useful Links

- [API Documentation](http://localhost:8000/docs)
- [Dashboard](http://localhost:8501)
- [Prometheus](http://localhost:9090)
- [Grafana](http://localhost:3000)
- [System Status](http://localhost:8000/health)
