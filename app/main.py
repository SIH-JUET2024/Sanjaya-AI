from fastapi import FastAPI
from app.chat import chat_router
from app.document_upload import document_router
from app.badwords import badwords_router

app = FastAPI()

app.include_router(chat_router)
app.include_router(document_router)
app.include_router(badwords_router)

# uvicorn app.main:app --reload
