"""
Configuration module for environment variables and global settings.
Loads environment variables from a .env file and sets key parameters for the application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Number of top documents to retrieve in search
TOP_K = int(os.getenv("TOP_K", 3))

# Name of the embedding model to use for vector search
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# API key for Gemini (Google Generative AI)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")