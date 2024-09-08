import logging
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.typing_analysis import router as typing_analysis_router
from app.api.v1.music_recomondation_routes import app as music_recomondation_router
from app.api.v1.voice_analysis import app as voice_analysis_router
from app.core.config import settings
import uvicorn
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

# CORS settings: Specify allowed origins, methods, and headers
origins = [
    "http://localhost",           # Allow requests from localhost
    "http://localhost:8000",      # Allow requests from frontend app on localhost:3000
    "http://127.0.0.1:8000",      # Allow requests from 127.0.0.1 on port 3000
    "http://192.168.1.102:8000",     # Replace with your actual domain
]

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    # Origins that are allowed to communicate with this backend
    allow_origins=origins,
    allow_credentials=True,           # Allow credentials such as cookies to be included
    # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],
    # Allow all headers to be sent with the requests
    allow_headers=["*"],
)

# Common router for all API routes with a common prefix
common_router = APIRouter()

# Include individual routers in the common router
common_router.include_router(typing_analysis_router, prefix="/typing_analysis")
common_router.include_router(
    music_recomondation_router, prefix="/music_recomondation")
common_router.include_router(
    voice_analysis_router, prefix="/voice_analyse"
)

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
