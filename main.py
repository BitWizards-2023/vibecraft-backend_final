from fastapi import FastAPI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Server is running!"}

# Get the port from environment variables
port = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
