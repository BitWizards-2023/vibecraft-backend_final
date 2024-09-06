from fastapi import FastAPI
from api.routes.typing_analysis_routes import router as typing_analysis_router

app = FastAPI()

# Register API routes
app.include_router(typing_analysis_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Emotion Detection API"}
