# app/utils.py
import requests
import openai
import numpy as np
import fitz  # PyMuPDF for PDFs
import textract
import csv

def make_post_request(url: str, json_payload: dict):
    """Utility to send a POST request."""
    response = requests.post(url, json=json_payload)
    return response

# Utility function to extract text from different file types
def extract_text_from_file(file):
    if file.content_type == "application/pdf":
        # Process PDF using PyMuPDF
        doc = fitz.open(file.file)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Process DOCX using textract
        return textract.process(file.file).decode('utf-8')

    elif file.content_type == "text/csv":
        # Process CSV
        text = ""
        with open(file.file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                text += " ".join(row) + "\n"
        return text

    else:
        raise ValueError("Unsupported file type.")

# Function to generate vector embeddings using OpenAI
def embed_text(text: str):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Cosine similarity calculation
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
