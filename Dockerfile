# Use an official Python runtime as a parent image
FROM python:3.12.1
 
# Set the working directory in the container
WORKDIR /app
 
# Install system-level dependencies required by librosa and ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
 
# Install librosa globally using pip
RUN pip install --no-cache-dir librosa
 
# Copy the current directory contents into the container
COPY . /app
 
# Install any remaining Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
 
# Expose port 8000 for the FastAPI app
EXPOSE 8000
 
# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]