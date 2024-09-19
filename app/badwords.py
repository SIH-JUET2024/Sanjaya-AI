from fastapi import APIRouter, HTTPException
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

badwords_router = APIRouter()

# Retrieve URLs from environment variables
ADD_BADWORD_URL = os.getenv("ADD_BADWORD_URL")
DELETE_BADWORD_URL = os.getenv("DELETE_BADWORD_URL")
GET_BADWORDS_URL = os.getenv("GET_BADWORDS_URL")

@badwords_router.post("/add-badword")
async def add_badword(word: str):
    try:
        payload = {"word": word}
        response = requests.post(ADD_BADWORD_URL, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to add bad word.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@badwords_router.get("/badwords")
async def get_badwords():
    try:
        response = requests.get(GET_BADWORDS_URL)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to get bad words.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@badwords_router.post("/delete-badword")
async def delete_badword(word: str):
    try:
        # First get the list of bad words to find the correct id
        badwords_response = requests.get(GET_BADWORDS_URL)
        if badwords_response.status_code == 200:
            badwords = badwords_response.json()
            # Find the bad word with the given word
            badword = next((bw for bw in badwords if bw['word'] == word), None)
            if not badword:
                raise HTTPException(status_code=404, detail="Bad word not found.")
            
            # Now delete the bad word by id
            payload = {"id": badword["id"]}
            delete_response = requests.post(DELETE_BADWORD_URL, json=payload)

            if delete_response.status_code == 200:
                return delete_response.json()
            else:
                raise HTTPException(status_code=delete_response.status_code, detail="Failed to delete bad word.")
        else:
            raise HTTPException(status_code=badwords_response.status_code, detail="Failed to fetch bad words.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
