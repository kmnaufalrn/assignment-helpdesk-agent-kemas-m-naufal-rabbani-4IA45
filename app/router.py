"""
Router — menentukan apakah query user butuh RAG (informasi/kebijakan)
atau tool call (aksi spesifik, misal cek status tiket).

Pola ini mengikuti workflow router yang sudah dipelajari di Sesi 12.4:
  user input -> intent detection -> jika butuh data -> retrieval (RAG)
                                  -> jika butuh aksi -> tool call
"""

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from google import genai

from app.rag import answer_from_knowledge_base
from app.tools import get_ticket_status

load_dotenv()

GENERATION_MODEL = "gemini-3.5-flash-lite"
TICKET_ID_PATTERN = re.compile(r"\bTICKET-\d{4}\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY tidak ditemukan. Isi variabel tersebut di file .env."
        )
    return genai.Client(api_key=api_key)


async def detect_intent(query: str) -> str:
    """
    Menentukan intent dari query user. Harus mengembalikan salah satu:
      - "faq"           -> pertanyaan informasi/kebijakan, arahkan ke RAG
      - "ticket_status"  -> permintaan cek status tiket, arahkan ke tool call
      - "unknown"        -> permintaan di luar cakupan atau tidak jelas
    """
    if not query or not query.strip():
        raise ValueError("Query tidak boleh kosong.")

    if TICKET_ID_PATTERN.search(query):
        return "ticket_status"

    prompt = f"""Klasifikasikan permintaan pengguna ke tepat satu intent:
- ticket_status: pengguna meminta status atau perkembangan tiket IT yang sudah ada.
- faq: pertanyaan informasi umum atau kebijakan IT/helpdesk.
- unknown: permintaan yang tidak berkaitan dengan IT/helpdesk atau tidak jelas.

Balas hanya dengan salah satu label: faq, ticket_status, atau unknown.

Permintaan pengguna: {query}"""
    response = _get_client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    label = (response.text or "").strip().lower().strip("`\"' .,:;")
    if label not in {"faq", "ticket_status", "unknown"}:
        raise RuntimeError(f"Gemini mengembalikan intent yang tidak valid: {label!r}")
    return label


async def handle_query(query: str) -> dict:
    """
    Fungsi utama yang dipanggil oleh endpoint /ask.
    Menjalankan intent detection, lalu merutekan ke RAG atau tool call yang sesuai.
    """
    intent = await detect_intent(query)

    if intent == "ticket_status":
        ticket_id = extract_ticket_id(query)
        if not ticket_id:
            return {
                "status": "error",
                "intent": "ticket_status",
                "data": "Sertakan nomor tiket dengan format TICKET-0001.",
            }
        data = get_ticket_status(ticket_id)
        return {"status": "success", "intent": "ticket_status", "data": data}

    elif intent == "faq":
        data = await answer_from_knowledge_base(query)
        return {"status": "success", "intent": "faq", "data": data}

    else:
        return {"status": "error", "intent": "unknown", "data": "Maaf, saya tidak memahami permintaan Anda."}


def extract_ticket_id(query: str) -> str:
    """
    Ekstrak ID tiket dengan format TICKET-XXXX dari query.
    """
    match = TICKET_ID_PATTERN.search(query or "")
    return match.group(0).upper() if match else ""
