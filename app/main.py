"""
Entry point aplikasi FastAPI — IT Helpdesk Assistant.

Alur endpoint utama:
  1. Menerima pertanyaan/permintaan dari user
  2. Meneruskan ke router (router.py) untuk menentukan intent
  3. Router memanggil RAG (rag.py) atau tool (tools.py) sesuai intent
  4. Mengembalikan response terstruktur ke client
"""

import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.router import handle_query

app = FastAPI(title="IT Helpdesk Assistant API")


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    status: str
    intent: str
    data: str


@app.get("/health")
async def health():
    """Endpoint sederhana untuk memastikan service berjalan."""
    return {"status": "ok"}


@app.get("/ask", include_in_schema=False)
async def ask_docs():
    return RedirectResponse(url="/docs")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Endpoint utama. Menerima pertanyaan bebas dari user (bisa berupa
    pertanyaan informasi/kebijakan ATAU permintaan cek status tiket),
    lalu meneruskannya ke router untuk diproses.

    """
    try:
        result = await handle_query(request.query)
        return AskResponse(**result)
    except Exception:
        logging.exception("Gagal memproses permintaan /ask.")
        return AskResponse(
            status="error",
            intent="error",
            data="Permintaan belum dapat diproses. Periksa konfigurasi dan coba lagi.",
        )
