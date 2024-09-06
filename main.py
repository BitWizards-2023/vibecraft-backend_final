import logging
import time
from fastapi import FastAPI
from app.api.v1.typing_analysis import router as typing_analysis_router
from app.core.config import settings
import uvicorn
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

# Include the router for typing analysis
app.include_router(typing_analysis_router, prefix="/api/v1")

# Background function to check server health
def log_health_check():
    while True:
        logger.info("Server is running and healthy.")
        time.sleep(30)  # Log every 30 seconds

# Start the health check logging in a separate thread
def start_health_check():
    health_thread = threading.Thread(target=log_health_check)
    health_thread.daemon = True  # Daemonize thread to ensure it shuts down with the app
    health_thread.start()

if __name__ == "__main__":
    # Start health check logging before running the server
    start_health_check()
    # Start the Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
