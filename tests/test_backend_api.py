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

    def test_cors_allows_vercel_production_origin(self) -> None:
        response = self.client.get("/health", headers={"Origin": "https://speechlabarnob.vercel.app"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "https://speechlabarnob.vercel.app")
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")

    def test_cors_preflight_options(self) -> None:
        response = self.client.options(
            "/health",
            headers={
                "Origin": "https://speechlabarnob.vercel.app",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "https://speechlabarnob.vercel.app")
        methods = response.headers.get("access-control-allow-methods", "")
        self.assertIn("GET", methods)

    def test_cors_allows_preview_branch_domains(self) -> None:
        response = self.client.get("/health", headers={"Origin": "https://speechlabarnob-preview-xyz.vercel.app"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "https://speechlabarnob-preview-xyz.vercel.app")

    def test_cors_rejects_unauthorized_origin(self) -> None:
        response = self.client.get("/health", headers={"Origin": "https://unauthorized-domain.com"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.headers.get("access-control-allow-origin"))


if __name__ == "__main__":
    unittest.main()
