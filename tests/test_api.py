from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from helpers.config import Settings


@pytest.fixture(autouse=True)
def disable_external_observability_for_api_tests(monkeypatch) -> None:
    """Keep API unit tests independent of the developer's local .env/services."""
    monkeypatch.setattr("api.app.get_settings", lambda: Settings(_env_file=None))


class FakeRAG:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.closed = False
        self.initialize_calls = 0
        self.indexed_count = 7

    def initialize(self) -> int:
        self.initialize_calls += 1
        return self.indexed_count

    def health(self) -> int:
        return self.indexed_count

    def query(self, question: str) -> dict:
        self.asked.append(question)
        return {
            "answer": "The answer is supported by Article 147.",
            "references": [
                {
                    "citation": "Egyptian Civil Code, Article 147",
                    "chunk_id": "internal-chunk-id",
                },
                {"citation": "Egyptian Civil Code, Article 147"},
                {"citation": "internal-chunk-id"},
            ],
        }

    def close(self) -> None:
        self.closed = True


def test_ask_returns_answer_and_unique_article_citations() -> None:
    rag = FakeRAG()
    app = create_app(rag_factory=lambda: rag)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "  ما آثار العقد؟  "})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The answer is supported by Article 147.",
        "sources": ["Egyptian Civil Code, Article 147"],
    }
    assert rag.asked == ["ما آثار العقد؟"]
    assert rag.initialize_calls == 1
    assert rag.closed


def test_root_serves_web_interface_and_static_assets() -> None:
    rag = FakeRAG()
    app = create_app(rag_factory=lambda: rag)

    with TestClient(app) as client:
        page = client.get("/")
        script = client.get("/static/app.js")
        styles = client.get("/static/styles.css")

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert 'id="question-form"' in page.text
    assert script.status_code == 200
    assert 'fetch("/ask"' in script.text
    assert styles.status_code == 200
    assert ".question-card" in styles.text
    assert rag.closed


def test_metrics_endpoint_exposes_bounded_http_request_metrics() -> None:
    rag = FakeRAG()
    app = create_app(rag_factory=lambda: rag)

    with TestClient(app) as client:
        client.post("/ask", json={"question": "ما آثار العقد؟"})
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert 'method="POST",route="/ask",status="200"' in response.text


def test_health_reports_ready_index_count_without_extra_errors() -> None:
    rag = FakeRAG()
    app = create_app(rag_factory=lambda: rag)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "documents_indexed": 7}
    assert rag.initialize_calls == 1
    assert rag.closed


def test_health_reports_initialization_errors_and_ask_returns_503() -> None:
    class FailingRAG(FakeRAG):
        def initialize(self) -> int:
            self.initialize_calls += 1
            raise RuntimeError("embedding model could not be loaded")

    rag = FailingRAG()
    app = create_app(rag_factory=lambda: rag)

    with TestClient(app) as client:
        health_response = client.get("/health")
        ask_response = client.post("/ask", json={"question": "ما آثار العقد؟"})

    assert health_response.status_code == 503
    assert health_response.json() == {
        "status": "unhealthy",
        "documents_indexed": 0,
        "errors": ["RuntimeError: embedding model could not be loaded"],
    }
    assert ask_response.status_code == 503
    assert rag.closed


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_ask_rejects_empty_or_whitespace_question(question: str) -> None:
    rag = FakeRAG()
    app = create_app(rag_factory=lambda: rag)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": question})

    assert response.status_code == 422
    assert rag.asked == []
    assert rag.closed
