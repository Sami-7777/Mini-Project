#!/usr/bin/env python3
"""
System startup script for the cyberattack detection system.
"""
import asyncio
import sys
import os
from pathlib import Path
import subprocess
import time
import signal

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import settings
from core.logger import logger
from database.connection import db_manager
from models.model_manager import model_manager
from monitoring.metrics import metrics_collector


class SystemManager:
    """Manages the cyberattack detection system startup and shutdown."""
    
    def __init__(self):
        self.processes = []
        self.shutdown_event = asyncio.Event()
    
    async def start_system(self):
        """Start the entire system."""
        try:
            logger.info("Starting cyberattack detection system")
            
            # Initialize database
            await self._initialize_database()
            
            # Initialize models
            await self._initialize_models()
            
            # Start monitoring
            await self._start_monitoring()
            
            # Start API server
            await self._start_api_server()
            
            # Start dashboard
            await self._start_dashboard()
            
            logger.info("System started successfully")
            
            # Wait for shutdown signal
            await self._wait_for_shutdown()
            
        except Exception as e:
            logger.error("Error starting system", error=str(e))
            raise
        finally:
            await self._shutdown_system()
    
    async def _initialize_database(self):
        """Initialize database connections."""
        try:
            logger.info("Initializing database connections")
            await db_manager.connect()
            logger.info("Database connections established")
            
        except Exception as e:
            logger.error("Error initializing database", error=str(e))
            raise
    
    async def _initialize_models(self):
        """Initialize ML models."""
        try:
            logger.info("Initializing ML models")
            await model_manager.initialize()
            logger.info("ML models initialized")
            
        except Exception as e:
            logger.error("Error initializing models", error=str(e))
            raise
    
    async def _start_monitoring(self):
        """Start monitoring services."""
        try:
            logger.info("Starting monitoring services")
            
            # Start Prometheus metrics server
            await metrics_collector.start_prometheus_server()
            
            logger.info("Monitoring services started")
            
        except Exception as e:
            logger.error("Error starting monitoring", error=str(e))
            raise
    
    async def _start_api_server(self):
        """Start the FastAPI server."""
        try:
            logger.info("Starting API server")
            
            # Start API server in background
            api_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "src.api.main:app",
                "--host", settings.api_host,
                "--port", str(settings.api_port),
                "--workers", str(settings.api_workers)
            ])
            
            self.processes.append(("API Server", api_process))
            
            # Wait for server to start
            await self._wait_for_service(f"http://{settings.api_host}:{settings.api_port}/health")
            
            logger.info("API server started", port=settings.api_port)
            
        except Exception as e:
            logger.error("Error starting API server", error=str(e))
            raise
    
    async def _start_dashboard(self):
        """Start the Streamlit dashboard."""
        try:
            logger.info("Starting dashboard")
            
            # Start dashboard in background
            dashboard_process = subprocess.Popen([
                sys.executable, "-m", "streamlit",
                "run", "src/dashboard/main.py",
                "--server.port", str(settings.dashboard_port),
                "--server.address", settings.dashboard_host,
                "--server.headless", "true"
            ])
            
            self.processes.append(("Dashboard", dashboard_process))
            
            # Wait for dashboard to start
            await self._wait_for_service(f"http://{settings.dashboard_host}:{settings.dashboard_port}")
            
            logger.info("Dashboard started", port=settings.dashboard_port)
            
        except Exception as e:
            logger.error("Error starting dashboard", error=str(e))
            raise
    
    async def _wait_for_service(self, url: str, timeout: int = 30):
        """Wait for a service to become available."""
        import aiohttp
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return
            except Exception:
                pass
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"Service {url} did not become available within {timeout} seconds")
    
    async def _wait_for_shutdown(self):
        """Wait for shutdown signal."""
        # Set up signal handlers
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received", signal=signum)
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for shutdown event
        await self.shutdown_event.wait()
    
    async def _shutdown_system(self):
        """Shutdown the system."""
        try:
            logger.info("Shutting down system")
            
            # Stop all processes
            for name, process in self.processes:
                try:
                    logger.info("Stopping process", name=name)
                    process.terminate()
                    
                    # Wait for process to terminate
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning("Process did not terminate gracefully", name=name)
                        process.kill()
                        process.wait()
                    
                    logger.info("Process stopped", name=name)
                    
                except Exception as e:
                    logger.error("Error stopping process", name=name, error=str(e))
            
            # Close database connections
            await db_manager.disconnect()
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))


async def main():
    """Main function."""
    system_manager = SystemManager()
    await system_manager.start_system()


if __name__ == "__main__":
    asyncio.run(main())
