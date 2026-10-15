"""Tests for FastAPI backend root and health endpoints."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend_api.main import app


class BackendApiTests(unittest.TestCase):
    """Test suite for top-level API routes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_root_endpoint_returns_ok_and_service_info(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload.get("service"), "Speech Separation and Diarization API")
        self.assertEqual(payload.get("docs"), "/docs")

    def test_health_endpoint_returns_ok(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_docs_endpoint_is_accessible(self) -> None:
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
