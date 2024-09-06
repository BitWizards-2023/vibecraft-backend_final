import logging
from fastapi import FastAPI, APIRouter
from app.api.v1.typing_analysis import router as typing_analysis_router
from app.core.config import settings
import uvicorn
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

# Common router for all API routes with a common prefix
common_router = APIRouter()

# Include individual routers in the common router
common_router.include_router(typing_analysis_router, prefix="/typing_analysis")

# Include the common router in the FastAPI app with the common prefix "/api/v1"
app.include_router(common_router, prefix="/api/v1")

# Health check endpoint (Make sure to attach the router with the health check to the app)
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "Server is running and healthy"}

# Background function to log server health
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
