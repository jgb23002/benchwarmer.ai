"""Frontend API regressions; all AI work is mocked and data stays in tmp_path."""

import base64
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Importing the app must not read the developer's real API credentials.
with patch("dotenv.load_dotenv", return_value=False):
    import server

from benchwarmer import chat_api, database


class FakeAgent:
    instances = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.state = SimpleNamespace(
            algo_specs=[], preferred_algo_spec_indices=None, algorithms=[],
            custom_algo_name=None, last_plot_path=None, instance_source=None,
            preferred_instance_source=None, config=None,
        )
        self.cleaned = False
        self.instances.append(self)

    def run_one_turn(self, messages, user_message, progress_cb):
        progress_cb({"type": "thinking"})
        if user_message == "fail":
            raise RuntimeError("simulated provider failure")
        if user_message == "choices":
            progress_cb({
                "type": "choice_prompt", "id": "instance_source",
                "title": "Choose instances", "multi_select": False,
                "options": [{"value": "generators", "label": "Generators"}],
            })
        reply = f"Reply to: {user_message}"
        return messages + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ], reply

    def _cleanup(self):
        self.cleaned = True


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "chat.db"))
    monkeypatch.setattr(chat_api, "OrchestratorAgent", FakeAgent)
    FakeAgent.instances = []
    chat_api._SESSIONS.clear()
    with TestClient(server.app) as test_client:
        yield test_client
    for session in chat_api._SESSIONS.values():
        shutil.rmtree(session.get("temp_dir", ""), ignore_errors=True)
    chat_api._SESSIONS.clear()


def events(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ")]


def start_chat(client, message="hello", **kwargs):
    stream = events(client.post("/api/chat", json={"message": message, **kwargs}))
    sid = next(event["session_id"] for event in stream if event["type"] == "session_start")
    return sid, stream


def test_routes_and_database_are_ready_on_startup(client):
    assert client.get("/api/sessions").json() == []
    assert client.get("/api/algorithms").json() == []
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/chat" in paths
    assert "/api/session/start" in paths  # Preserve the existing API.


@pytest.mark.parametrize("origin", [
    "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000",
])
def test_frontend_cors_preflight_is_allowed(client, origin):
    response = client.options("/api/chat", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_other_origins_are_not_allowed(client):
    response = client.options("/api/chat", headers={
        "Origin": "https://unrelated.example",
        "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_streamed_reply_is_saved_and_conversation_continues(client):
    sid, stream = start_chat(client)
    assert stream[-1]["type"] == "done"
    assert stream[-1]["reply"] == "Reply to: hello"
    assert client.get(f"/api/chat/{sid}/status").json() == {"processing": False}
    history = client.get(f"/api/chat/{sid}").json()
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert client.get("/api/sessions").json()[0]["id"] == sid

    _, second = start_chat(client, "continue", session_id=sid)
    assert second[-1]["reply"] == "Reply to: continue"
    assert len(client.get(f"/api/chat/{sid}").json()) == 4
    assert len(FakeAgent.instances) == 1


def test_interactive_metadata_survives_history_reload(client):
    sid, stream = start_chat(client, "choices")
    assert any(event["type"] == "choice_prompt" for event in stream)
    assistant = client.get(f"/api/chat/{sid}").json()[-1]
    assert assistant["choicePrompt"]["id"] == "instance_source"
    assert assistant["choicePrompt"]["options"][0]["value"] == "generators"


def test_provider_error_ends_stream_and_releases_session(client):
    sid, stream = start_chat(client, "fail")
    assert stream[-1]["type"] == "error"
    assert "simulated provider failure" in stream[-1]["error"]
    assert client.get(f"/api/chat/{sid}/status").json() == {"processing": False}
    assert "simulated provider failure" in client.get(f"/api/chat/{sid}").json()[-1]["content"]


def test_sessions_can_be_renamed_and_deleted(client):
    sid, _ = start_chat(client)
    assert client.patch(f"/api/sessions/{sid}", json={"title": "My experiment"}).status_code == 200
    assert client.get("/api/sessions").json()[0]["title"] == "My experiment"
    assert client.patch(f"/api/sessions/{sid}", json={"title": " "}).status_code == 400
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get("/api/sessions").json() == []
    assert client.get(f"/api/chat/{sid}").json() == []
    assert FakeAgent.instances[0].cleaned


def test_algorithm_code_can_be_loaded_by_frontend(client):
    database.save_algorithm("example", "class Example: pass")
    assert client.get("/api/algorithms").json()[0]["name"] == "example"
    assert client.get("/api/algorithms/example").json()["code"] == "class Example: pass"
    assert client.get("/api/algorithms/missing").status_code == 404


def test_uploads_and_model_selection_reach_the_agent(client):
    raw_pdf = b"%PDF-1.4 sample"
    raw_code = b"class Example: pass"
    sid, _ = start_chat(
        client, "compare", execution_mode="modal", llm_backend="nemotron",
        pdfs=[{"filename": "../paper.pdf", "content_base64": base64.b64encode(raw_pdf).decode()}],
        custom_algorithm_file={"filename": "example.py", "content_base64": base64.b64encode(raw_code).decode()},
    )
    options = FakeAgent.instances[0].options
    assert options["execution_mode"] == "modal"
    assert options["intake_backend"] == "nemotron"
    assert Path(options["pdf_paths"][0]).read_bytes() == raw_pdf
    assert Path(options["custom_algo_path"]).read_bytes() == raw_code
    assert Path(options["pdf_paths"][0]).is_relative_to(chat_api._SESSIONS[sid]["temp_dir"])


@pytest.mark.parametrize("attachment", [
    {"filename": "wrong.exe", "content_base64": "YQ=="},
    {"filename": "test.py", "content_base64": "not valid base64!"},
    {"filename": "test.py", "content_base64": ""},
    {"filename": "test.py", "content_base64": base64.b64encode(b"x" * (512 * 1024 + 1)).decode()},
])
def test_invalid_uploads_are_rejected_before_ai_work(client, attachment):
    response = client.post("/api/chat", json={"message": "test", "custom_algorithm_file": attachment})
    assert response.status_code == 400
    assert FakeAgent.instances == []


def test_expired_and_busy_sessions_report_errors_without_ai_calls(client):
    expired = events(client.post("/api/chat", json={"session_id": "expired", "message": "hello"}))
    assert expired[-1]["type"] == "error"
    chat_api._SESSIONS["busy"] = {"processing": True}
    busy = events(client.post("/api/chat", json={"session_id": "busy", "message": "hello"}))
    assert "still being processed" in busy[-1]["error"]
    assert client.delete("/api/sessions/busy").status_code == 409
    assert FakeAgent.instances == []


def test_blank_chat_is_rejected(client):
    assert client.post("/api/chat", json={"message": " "}).status_code == 400
    assert FakeAgent.instances == []
