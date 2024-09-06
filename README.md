## Overview

This project provides an API for detecting emotions using two methods:
1. **Text-based Emotion Detection**: Uses the OpenAI API to analyze the emotion from a meaningful sentence or text.
2. **Typing Dynamics-based Emotion Detection**: Uses a pre-trained machine learning model (XGBoost) to detect emotions based on typing dynamics (e.g., key press timings).

### Key Features:
- **Real-time Emotion Detection**: Detects emotions based on user input.
- **OpenAI Integration**: Uses OpenAI's GPT models for analyzing text input.
- **Machine Learning Model**: Analyzes typing patterns like `D1U1`, `D1D2`, and `typing speed` to detect emotions.
- **RESTful API**: Built with **FastAPI**, supporting both JSON and form-encoded input.
- **Dockerized**: Containerized using Docker for easy deployment.

---

## Table of Contents
1. [Installation](#installation)
2. [API Endpoints](#api-endpoints)
3. [Project Structure](#project-structure)
4. [Environment Variables](#environment-variables)
5. [Run the Application](#run-the-application)
6. [Testing the API](#testing-the-api)
7. [Deployment](#deployment)
8. [License](#license)

---

## Installation

### Prerequisites:
- **Python 3.7+**
- **Virtual Environment (optional but recommended)**
- **Docker** (for containerization)
- **OpenAI API Key** (for text-based emotion detection)
