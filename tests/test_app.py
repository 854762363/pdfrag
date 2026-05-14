"""
Tests for FastAPI app startup and health endpoint.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


class TestAppHealth:
    """Test basic application health."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self) -> None:
        """Verify /health returns OK."""
        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "version" in data

    @pytest.mark.asyncio
    async def test_app_title(self) -> None:
        """Verify app metadata."""
        from src.main import app
        assert app.title == "PDFRAG"
