from fastapi import FastAPI
from app.chat import chat_router
from app.document_upload import document_router
from app.badwords import badwords_router

# Initialize FastAPI app
app = FastAPI()

# Include the routers for different functionalities
app.include_router(chat_router)
app.include_router(document_router)
app.include_router(badwords_router)

# Run with `uvicorn app.main:app --reload`
