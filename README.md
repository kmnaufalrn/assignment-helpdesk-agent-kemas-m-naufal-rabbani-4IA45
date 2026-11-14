# IT Helpdesk Assistant

Prototype backend AI Helpdesk untuk menjawab FAQ IT menggunakan RAG dan memeriksa
status tiket melalui tool call. API dibangun dengan FastAPI dan dapat dijalankan
secara lokal atau di dalam Docker.

## Problem Statement dan Tujuan

Tim People & IT Operations menerima dua jenis permintaan: pertanyaan kebijakan
berulang dan permintaan spesifik untuk mengecek status tiket. Aplikasi ini
mengarahkan pertanyaan FAQ ke knowledge base dan permintaan status tiket ke data
tiket, sehingga pencarian tiket tidak bergantung pada semantic search.

Tujuan aplikasi:

- Mengambil informasi relevan dari knowledge base dan menyusun jawaban berbasis
  konteks melalui Gemini.
- Mendeteksi intent faq, ticket_status, atau unknown.
- Menyediakan endpoint FastAPI dan menjalankannya  di Docker.

## Asumsi dan Workflow

- Knowledge base dan daftar tiket menggunakan CSV yang tersedia di folder data/.
- ID tiket mengikuti format TICKET-XXXX
- FAQ dijawab hanya berdasarkan konteks yang ditemukan. Jika pertanyaan di luar
  cakupan, sistem a menyatakan bahwa informasi tidak tersedia.
- Gemini memerlukan `GEMINI_API_KEY` dari environment

Alur request:


query masuk
  -> deteksi intent
  -> faq: embedding query -> pencarian FAISS -> prompt berkonteks -> jawaban Gemini
  -> ticket_status: ekstrak ticket_id -> cari data CSV
  -> unknown: respons bahwa permintaan tidak dipahami/di luar cakupan
  -> response JSON 


## Struktur Project

```text
app/
├── main.py      
├── router.py     
├── rag.py       
└── tools.py      
data/
├── it_helpdesk_knowledge_base.csv
└── mock_tickets.csv
tests/
└── test_helpdesk.py
Dockerfile
requirements.txt
.env.example
```

## Menjalankan Secara Lokal

PowerShell:

```powershell
.env.example .env
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```
 `http://localhost:8000/docs`.


## Menjalankan dengan Docker

Pastikan Docker Desktop berjalan dan `.env` berisi `GEMINI_API_KEY`.

```powershell
docker build -t it-helpdesk-assistant .
docker run --rm -p 8000:8000 --env-file .env it-helpdesk-assistant
```

## Pengujian

Jalankan automated tests:

```powershell
python -m unittest discover -s tests -v
```

Lima skenario endpoint `/ask` yang diuji:

| Skenario | Query | Intent/hasil yang diharapkan | 
|---|---|---|---|
| FAQ cuti | `Berapa jatah cuti tahunan karyawan tetap?` | `faq`, jawaban menyebut 12 hari | 
| FAQ VPN | `Bagaimana cara reset password VPN?` | `faq`, jawaban tentang portal self-service | 
| Tiket tersedia | `Cek status TICKET-0001` | `ticket_status`, status `Selesai` | 
| Tiket tidak tersedia | `Cek status TICKET-9999` | `ticket_status`, tiket tidak ditemukan | 
| Di luar cakupan | `Siapa juara Piala Dunia?` | `unknown`, permintaan tidak dipahami | Tidak 


Contoh request manual:

```powershell
curl.exe -X POST http://localhost:8000/ask `
 
```


