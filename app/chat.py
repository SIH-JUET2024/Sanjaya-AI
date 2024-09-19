from fastapi import APIRouter, HTTPException
from typing import List
import requests
import openai
from app.utils import extract_text_from_file, embed_text, cosine_similarity
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DocxLoader, CSVLoader

# Load environment variables from .env file
load_dotenv()

chat_router = APIRouter()

# API endpoints for bad words
GET_BADWORDS_URL = os.getenv("GET_BADWORDS_URL")
ADD_FLAGGED_WORD_URL = os.getenv("ADD_BADWORD_URL")

# API endpoints for files
FETCH_FILES_URL = os.getenv("GET_FILES_URL")

# OpenAI API key (ensure it's stored in a safe environment)
openai.api_key = os.getenv("OPENAI_API_KEY")
# It has been stored in two variables due to an error.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector store (simple in-memory storage for this example)
vector_store = []

# Get bad words list from API
def get_bad_words() -> List[str]:
    response = requests.get(GET_BADWORDS_URL)
    if response.status_code == 200:
        badwords_data = response.json()
        return [word['word'] for word in badwords_data]
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve bad words.")

# Add flagged bad word to API
def flag_bad_word(word: str):
    payload = {"word": word}
    response = requests.post(ADD_FLAGGED_WORD_URL, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to flag bad word.")

# Function to check and censor bad words
def censor_bad_words(user_input: str) -> str:
    bad_words = get_bad_words()
    censored_input = user_input
    flagged_words = []

    for word in bad_words:
        if word.lower() in user_input.lower():
            censored_input = censored_input.replace(word, "***")
            flagged_words.append(word)

    # Flag bad words found in the user input
    for word in flagged_words:
        flag_bad_word(word)

    return censored_input

# Function to interact with OpenAI API
def get_openai_response(prompt: str) -> str:
    response = openai.Completion.create(
        model="gpt-3.5",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to find the best matching document text using vector similarity
def query_documents(query: str):
    if not vector_store:
        raise HTTPException(status_code=404, detail="No documents uploaded for querying.")
    
    query_embedding = embed_text(query)
    best_match = None
    best_score = -1

    for doc in vector_store:
        similarity_score = cosine_similarity(query_embedding, doc['embedding'])
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = doc

    return best_match['text'] if best_match else None

# Chatbot endpoint
@chat_router.post("/chat")
async def chat_with_bot(user_input: str):
    global vector_store
    try:
        # Censor bad words and flag them
        censored_input = censor_bad_words(user_input)

        # Check if the user is querying a document
        document_response = query_documents(censored_input)

        if document_response:
            prompt = f"User query: {censored_input}. Relevant document content: {document_response}. Provide an answer based on this information."
        else:
            prompt = f"User query: {censored_input}. Provide a helpful response."

        # Get response from OpenAI
        openai_response = get_openai_response(prompt)

        # If no vector store is available, process the latest file
        if vector_store is None:
            response = requests.get(FETCH_FILES_URL)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch files.")

            files = response.json()
            if not files:
                raise HTTPException(status_code=404, detail="No files found.")

            # Get the file with the highest ID
            latest_file = max(files, key=lambda x: x['id'])
            file_url = latest_file['url']
            file_type = latest_file['type']

            # Download the file content
            file_content = requests.get(file_url).content

            # Process the file based on its type (PDF, DOCX, CSV)
            if file_type == "application/pdf":
                loader = PyPDFLoader(file_content)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                loader = DocxLoader(file_content)
            elif file_type == "text/csv":
                loader = CSVLoader(file_content)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format.")

            # Load and split the text
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)

            # Create embeddings and store them in a vector store for fast querying
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            vector_store = FAISS.from_documents(docs, embeddings)

        # Query the vector store for relevant document snippets based on the user's question
        docs = vector_store.similarity_search(censored_input, k=5)  # Get the top 5 most relevant results

        # Format the retrieved document snippets and send the query to OpenAI for a response
        context = "\n".join([doc.page_content for doc in docs])

        # Use OpenAI to generate the chatbot response based on the context
        response = openai.Completion.create(
            engine="gpt-3.5",
            prompt=f"Context: {context}\n\nUser Query: {censored_input}\n\nProvide a detailed response:",
            max_tokens=300,
            temperature=0.5,
        )

        answer = response.choices[0].text.strip()
        return {"response": openai_response, "flagged": censored_input != user_input, "document_answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))