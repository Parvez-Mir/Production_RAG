from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_parses_uploaded_text_file() -> None:
    response = client.post(
        "/api/ingest",
        files={"file": ("notes.txt", b"First line\nSecond line", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "parsed"
    assert response.json()["filename"] == "notes.txt"
    assert response.json()["content"] == "First line\nSecond line"
    assert response.json()["metadata"]["file_type"] == "txt"


def test_ingest_rejects_unsupported_file_type() -> None:
    response = client.post(
        "/api/ingest",
        files={"file": ("malware.exe", b"content", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_ingest_rejects_files_over_five_megabytes() -> None:
    response = client.post(
        "/api/ingest",
        files={"file": ("large.txt", b"x" * (5 * 1024 * 1024 + 1), "text/plain")},
    )

    assert response.status_code == 413
    assert "5 MB" in response.json()["detail"]
