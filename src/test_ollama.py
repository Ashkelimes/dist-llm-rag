import os
import httpx
import json
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "phi3:mini")

print(f"Testing local Ollama with model: {MODEL_NAME}...")
try:
    # Ollama's generate endpoint for single-turn prompts
    payload = {
        "model": MODEL_NAME,
        "prompt": "What is distributed computing? Answer in one sentence.",
        "stream": False
    }
    response = httpx.post(f"{BASE_URL}/api/generate", json=payload, timeout=60.0)
    response.raise_for_status()
    
    # Extract and print the AI's response
    reply = response.json().get("response", "No text returned.")
    print(f"AI Reply: {reply}")
except Exception as e:
    print(f"Error: {e}")
