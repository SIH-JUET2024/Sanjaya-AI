from fastapi import APIRouter, UploadFile, File, HTTPException
import requests
import os
from dotenv import load_dotenv

document_router = APIRouter()

# Retrieve URLs from environment variables
UPLOAD_URL = os.getenv("UPLOAD_URL")
GET_FILES_URL = os.getenv("GET_FILES_URL")
DELETE_FILE_URL = os.getenv("DELETE_FILE_URL")

@document_router.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Send a POST request to the upload file API endpoint with form data
        files = {'file': (file.filename, file.file, file.content_type)}
        response = requests.post(UPLOAD_URL, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to upload file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@document_router.get("/files")
async def get_files():
    try:
        response = requests.get(GET_FILES_URL)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch files.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@document_router.post("/delete-file")
async def delete_file(file_id: int):
    try:
        payload = {"id": file_id}
        response = requests.post(DELETE_FILE_URL, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to delete file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
