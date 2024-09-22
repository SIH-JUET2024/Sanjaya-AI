from fastapi import APIRouter, HTTPException
from typing import List
import requests
import openai
from app.utils import cosine_similarity
import os
import tempfile
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Load environment variables from .env file
load_dotenv()

chat_router = APIRouter()

# API endpoints and environment variables
GET_BADWORDS_URL = os.getenv("GET_BADWORDS_URL")
ADD_FLAGGED_WORD_URL = os.getenv("ADD_BADWORD_URL")
FETCH_FILES_URL = os.getenv("GET_FILES_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI API key
openai.api_key = OPENAI_API_KEY

vector_store = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    user_input: str

def embed_text(text: str):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    if isinstance(text, str):
        vector = embeddings.embed_query(text)
        return vector
    elif isinstance(text, list):
        vectors = [embeddings.embed_query(t) for t in text]
        return vectors
    else:
        raise ValueError("Input must be a string or a list of strings.")

def fetch_file_content(file_url: str, file_type: str):
    try:
        response = requests.get(file_url)
        response.raise_for_status()

        if file_type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()
                return tmp_file.name
        elif file_type == "text/csv":
            return response.content.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch file: {str(e)}")

def initialize_vector_store():
    global vector_store
    if vector_store is not None:
        return

    logger.info("Fetching files from external API")
    response = requests.get(FETCH_FILES_URL)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch files.")

    files = response.json()
    if not files:
        raise HTTPException(status_code=404, detail="No files found.")

    latest_file = max(files, key=lambda x: x['id'])
    file_url, file_type = latest_file['url'], latest_file['type']

    logger.info(f"Downloading file content from {file_url}")
    file_content = fetch_file_content(file_url, file_type)

    if file_type == "application/pdf":
        loader = PyPDFLoader(file_content)
    elif file_type == "text/csv":
        loader = CSVLoader(file_content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    logger.info("Loading and splitting the text")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    logger.info("Creating embeddings and initializing the vector store")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(docs, embeddings)

def get_bad_words() -> List[str]:
    response = requests.get(GET_BADWORDS_URL)
    if response.status_code == 200:
        badwords_data = response.json()
        return [word['word'] for word in badwords_data]
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve bad words.")

def flag_bad_word(word: str):
    payload = {"word": word}
    response = requests.post(ADD_FLAGGED_WORD_URL, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to flag bad word.")

def censor_bad_words(user_input: str) -> str:
    bad_words = get_bad_words()
    censored_input = user_input.lower()
    flagged_words = set()

    for word in bad_words:
        if word.lower() in censored_input:
            censored_input = censored_input.replace(word.lower(), "***")
            flagged_words.add(word)

    for word in flagged_words:
        flag_bad_word(word)

    return censored_input

def get_openai_response(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

def query_documents(query: str):
    if not vector_store:
        raise HTTPException(status_code=404, detail="No documents uploaded for querying.")
    
    query_embedding = embed_text(query)
    docs = vector_store.similarity_search(query_embedding, k=5)
    return docs[0].page_content if docs else None

@chat_router.post("/chat")
async def chat_with_bot(request: ChatRequest):
    try:
        censored_input = censor_bad_words(request.user_input)

        if vector_store is None:
            initialize_vector_store()

        document_response = query_documents(censored_input)
        prompt = f"User query: {censored_input}. Relevant document content: {document_response or 'No document found.'}. Provide an answer based on this information."
        openai_response = get_openai_response(prompt)

        return {
            "response": openai_response, 
            "flagged": censored_input != request.user_input.lower(), 
            "document_answer": document_response
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")