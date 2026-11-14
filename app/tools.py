"""
Modul Tools — fungsi-fungsi "aksi" yang dipanggil router ketika intent
BUKAN pertanyaan informasi/kebijakan, melainkan permintaan spesifik
yang butuh pengecekan data langsung (bukan retrieval semantik).

Pola ini mengikuti konsep Tool Integration di Sesi 12.3.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_TICKETS_PATH = PROJECT_ROOT / "data" / "mock_tickets.csv"


def get_ticket_status(ticket_id: str) -> str:
    """
    Mencari status tiket berdasarkan ticket_id pada data mock_tickets.csv.

    """
    ticket_id = (ticket_id or "").strip().upper()
    if not ticket_id:
        return "Sertakan nomor tiket dengan format TICKET-0001."
    if not MOCK_TICKETS_PATH.is_file():
        raise FileNotFoundError(f"Data tiket tidak ditemukan: {MOCK_TICKETS_PATH}")

    tickets = pd.read_csv(MOCK_TICKETS_PATH, dtype=str).fillna("")
    required_columns = {"ticket_id", "subject", "status", "last_update"}
    missing_columns = required_columns.difference(tickets.columns)
    if missing_columns:
        raise ValueError(
            "Kolom data tiket tidak lengkap: " + ", ".join(sorted(missing_columns))
        )

    matches = tickets[tickets["ticket_id"].str.strip().str.upper() == ticket_id]
    if matches.empty:
        return f"Tiket {ticket_id} tidak ditemukan."

    ticket = matches.iloc[0]
    return (
        f"Status tiket {ticket_id}:\n"
        f"Subjek: {ticket['subject']}\n"
        f"Status: {ticket['status']}\n"
        f"Pembaruan terakhir: {ticket['last_update']}"
    )
