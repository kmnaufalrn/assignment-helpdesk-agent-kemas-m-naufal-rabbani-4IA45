import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

from app import rag
from app.main import app
from app.router import detect_intent, extract_ticket_id, handle_query
from app.tools import get_ticket_status


class TicketToolTests(unittest.TestCase):
    def test_returns_ticket_details_for_existing_ticket(self):
        result = get_ticket_status("ticket-0001")

        self.assertIn("Status tiket TICKET-0001", result)
        self.assertIn("Laptop tidak bisa menyala", result)
        self.assertIn("Selesai", result)

    def test_returns_message_for_unknown_ticket(self):
        result = get_ticket_status("TICKET-9999")

        self.assertEqual(result, "Tiket TICKET-9999 tidak ditemukan.")


class RouterTests(unittest.IsolatedAsyncioTestCase):
    def test_extracts_ticket_id_case_insensitively(self):
        self.assertEqual(extract_ticket_id("cek ticket-0005 ya"), "TICKET-0005")
        self.assertEqual(extract_ticket_id("cek status tiket"), "")

    async def test_ticket_id_routes_directly_to_status_tool(self):
        result = await handle_query("Tolong cek TICKET-0001")

        self.assertEqual(result["intent"], "ticket_status")
        self.assertIn("Laptop tidak bisa menyala", result["data"])

    async def test_faq_intent_uses_gemini_classification(self):
        class FakeResponse:
            text = "faq"

        class FakeModels:
            def generate_content(self, **kwargs):
                return FakeResponse()

        class FakeClient:
            models = FakeModels()

        with patch("app.router._get_client", return_value=FakeClient()):
            result = await detect_intent("Bagaimana cara reset password VPN?")

        self.assertEqual(result, "faq")

    async def test_out_of_scope_intent_is_accepted(self):
        class FakeResponse:
            text = "unknown"

        class FakeModels:
            def generate_content(self, **kwargs):
                return FakeResponse()

        class FakeClient:
            models = FakeModels()

        with patch("app.router._get_client", return_value=FakeClient()):
            result = await detect_intent("Siapa juara Piala Dunia?")

        self.assertEqual(result, "unknown")


class ApiScenarioTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_browser_get_ask_redirects_to_api_docs(self):
        response = self.client.get("/ask", follow_redirects=False)

        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/docs")

    def test_faq_annual_leave_routes_to_rag(self):
        with (
            patch(
                "app.router.detect_intent",
                new=AsyncMock(return_value="faq"),
            ),
            patch(
                "app.router.answer_from_knowledge_base",
                new=AsyncMock(return_value="Karyawan tetap mendapat 12 hari cuti."),
            ) as answer,
        ):
            response = self.client.post(
                "/ask",
                json={"query": "Berapa jatah cuti tahunan karyawan tetap?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "faq")
        self.assertIn("12 hari", response.json()["data"])
        answer.assert_awaited_once()

    def test_faq_vpn_routes_to_rag(self):
        with (
            patch(
                "app.router.detect_intent",
                new=AsyncMock(return_value="faq"),
            ),
            patch(
                "app.router.answer_from_knowledge_base",
                new=AsyncMock(return_value="Reset melalui portal self-service IT."),
            ) as answer,
        ):
            response = self.client.post(
                "/ask",
                json={"query": "Bagaimana cara reset password VPN?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "faq")
        self.assertIn("self-service", response.json()["data"])
        answer.assert_awaited_once()

    def test_known_ticket_routes_to_tool(self):
        response = self.client.post(
            "/ask",
            json={"query": "Cek status TICKET-0001"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "ticket_status")
        self.assertIn("Laptop tidak bisa menyala", response.json()["data"])

    def test_unknown_ticket_returns_not_found_message(self):
        response = self.client.post(
            "/ask",
            json={"query": "Cek status TICKET-9999"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "ticket_status")
        self.assertIn("tidak ditemukan", response.json()["data"])

    def test_out_of_scope_query_returns_unknown_response(self):
        with patch(
            "app.router.detect_intent",
            new=AsyncMock(return_value="unknown"),
        ):
            response = self.client.post(
                "/ask",
                json={"query": "Siapa juara Piala Dunia?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "unknown")
        self.assertIn("tidak memahami", response.json()["data"])


class RagTests(unittest.TestCase):
    def setUp(self):
        self.previous_index = rag._index
        self.previous_documents = rag._documents
        self.previous_df = rag._df
        rag._index = None
        rag._documents = None
        rag._df = None

    def tearDown(self):
        rag._index = self.previous_index
        rag._documents = self.previous_documents
        rag._df = self.previous_df

    def test_builds_faiss_index_and_returns_top_matches(self):
        vector = np.array([1.0, 0.0], dtype=np.float32)
        with patch("app.rag._get_embedding", return_value=vector):
            results = rag.semantic_search("berapa jatah cuti?", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIn("question", results[0])
        self.assertIn("answer", results[0])
        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_answer_uses_retrieved_context(self):
        class FakeResponse:
            text = "Karyawan tetap mendapat 12 hari cuti."

        class FakeModels:
            def generate_content(self, **kwargs):
                self.prompt = kwargs["contents"]
                return FakeResponse()

        class FakeClient:
            models = FakeModels()

        document = {
            "faq_id": "1",
            "category": "Cuti & Absensi",
            "question": "Berapa jatah cuti?",
            "answer": "Karyawan tetap mendapat 12 hari cuti.",
            "score": 0.9,
        }
        with (
            patch("app.rag.semantic_search", return_value=[document]),
            patch("app.rag._get_client", return_value=FakeClient()),
        ):
            result = asyncio.run(rag.answer_from_knowledge_base("Berapa jatah cuti?"))

        self.assertEqual(result, FakeResponse.text)


if __name__ == "__main__":
    unittest.main()
