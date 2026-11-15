"""Chat, history, and algorithm endpoints used by the React frontend."""

import asyncio
import base64
import binascii
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from benchwarmer.agents.orchestrator import OrchestratorAgent

logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_app):
    init_db()
    yield

router = APIRouter(lifespan=_lifespan)

class FileAttachment(BaseModel):
    filename: str
    content_base64: str = Field(max_length=28 * 1024 * 1024)

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    execution_mode: Literal["local", "modal"] = "local"  # local | modal (first turn)
    llm_backend: Literal["claude", "nemotron"] = "claude"  # claude | nemotron (first turn)
    pdfs: Optional[List[FileAttachment]] = None
    custom_algorithm_file: Optional[FileAttachment] = None  # .py file

def _json_default(obj: Any) -> Any:
    """Custom JSON serializer for numpy/pandas types."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def _safe_json(obj: Any) -> str:
    """JSON-encode with numpy/pandas safety."""
    return json.dumps(obj, default=_json_default, ensure_ascii=False)

def _save_attachment(attachment, directory, extension, max_bytes):
    name = os.path.basename(attachment.filename)
    if not name.lower().endswith(extension):
        raise HTTPException(status_code=400, detail=f"Expected a {extension} attachment")
    try:
        raw = base64.b64decode(attachment.content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid attachment encoding") from exc
    if not raw or len(raw) > max_bytes:
        raise HTTPException(status_code=400, detail="Attachment is empty or exceeds the upload limit")
    path = os.path.join(directory, name)
    with open(path, "wb") as target:
        target.write(raw)
    return path

_SESSIONS: Dict[str, Dict[str, Any]] = {}  # session_id -> {agent, messages}

from benchwarmer.database import init_db, save_message, get_messages

@router.get("/api/chat/{session_id}")
async def get_chat_history(session_id: str):
    """Retrieve chat history for a given session."""
    messages = get_messages(session_id)
    if not messages:
        return []
    return messages

from benchwarmer.database import get_all_sessions, delete_session, rename_session

@router.get("/api/sessions")
async def get_sessions():
    """Retrieve all chat sessions."""
    return get_all_sessions()

@router.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a chat session and all its messages."""
    session = _SESSIONS.get(session_id)
    if session and session.get("processing"):
        raise HTTPException(status_code=409, detail="This session is still processing")
    delete_session(session_id)
    session = _SESSIONS.pop(session_id, None)
    if session:
        session["agent"]._cleanup()
        shutil.rmtree(session.get("temp_dir", ""), ignore_errors=True)
    return {"ok": True}

@router.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, body: dict):
    """Rename a chat session."""
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title required")
    rename_session(session_id, title)
    return {"ok": True}

from benchwarmer.database import get_all_algorithms, get_algorithm_code

@router.get("/api/algorithms")
async def list_algorithms():
    """Retrieve all generated algorithms."""
    return get_all_algorithms()

@router.get("/api/algorithms/{name}")
async def get_algorithm(name: str):
    """Retrieve code for a specific algorithm."""
    code = get_algorithm_code(name)
    if not code:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return {"name": name, "code": code}



# ─── Session processing status ────────────────────────────────────────────────

@router.get("/api/chat/{session_id}/status")
async def get_chat_status(session_id: str):
    """Check if the backend is still processing a turn for this session."""
    sess = _SESSIONS.get(session_id)
    if sess:
        return {"processing": sess.get("processing", False)}
    return {"processing": False}


# ─── SSE Chat Endpoint ───────────────────────────────────────────────────────

@router.post("/api/chat")
async def chat_turn(request: ChatRequest):
    """
    Multi-turn chat endpoint — mirrors the CLI orchestrator flow.

    The orchestrator runs in a **background thread** that always saves
    its result to the DB — even if the SSE client disconnects mid-stream
    (e.g. browser refresh).  The SSE generator simply reads from the
    shared event queue and forwards events to the client.

    Returns a **Server-Sent Events** stream with real-time progress:
        data: {"type": "thinking"}
        data: {"type": "tool_start", "tool": "run_intake", ...}
        data: {"type": "tool_end",   "tool": "run_intake", "result": "..."}
        data: {"type": "done", "session_id": "...", "reply": "...", "chart": ...}
        data: {"type": "error", "error": "..."}
    """
    session_id = request.session_id
    agent: Optional[OrchestratorAgent] = None
    messages: list = []

    # ── Session setup (synchronous, before streaming) ──────────────────────
    if session_id and session_id in _SESSIONS:
        sess = _SESSIONS[session_id]
        # If session is already processing a turn, reject the new request
        if sess.get("processing", False):
            async def busy_stream():
                yield f"data: {_safe_json({'type': 'error', 'session_id': session_id, 'error': 'A turn is still being processed. Please wait.'})}\n\n"
            return StreamingResponse(
                busy_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        agent = sess["agent"]
        messages = sess["messages"]
    elif session_id and session_id not in _SESSIONS:
        # Session was lost (e.g. backend restarted). Tell the frontend.
        async def expired_stream():
            yield f"data: {_safe_json({'type': 'error', 'session_id': session_id, 'error': 'Session expired — the backend was restarted. Please start a new conversation.'})}\n\n"
        return StreamingResponse(
            expired_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        if not request.message.strip() and not request.pdfs and not request.custom_algorithm_file:
            raise HTTPException(status_code=400, detail="A message or attachment is required")
        pdf_paths: list[str] = []
        custom_algo_path: Optional[str] = None
        temp_dir = tempfile.mkdtemp(prefix="benchwarmer_chat_")
        try:
            for pdf in request.pdfs or []:
                directory = tempfile.mkdtemp(dir=temp_dir)
                pdf_paths.append(_save_attachment(pdf, directory, ".pdf", 20 * 1024 * 1024))
            if request.custom_algorithm_file:
                custom_algo_path = _save_attachment(
                    request.custom_algorithm_file, temp_dir, ".py", 512 * 1024
                )
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        # Resolve LLM backend — "claude" or "nemotron"
        backend_choice = (request.llm_backend or "claude").lower()
        nemotron_url = os.environ.get("NEMOTRON_BASE_URL", "http://localhost:11434/v1")
        nemotron_model = os.environ.get(
            "NEMOTRON_MODEL",
            "hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_K_M",
        )

        agent = OrchestratorAgent(
            execution_mode=request.execution_mode or "local",
            intake_backend=backend_choice,
            pdf_paths=pdf_paths or None,
            custom_algo_path=custom_algo_path,
            nemotron_url=nemotron_url if backend_choice == "nemotron" else None,
            nemotron_model=nemotron_model if backend_choice == "nemotron" else None,
        )
        logger.info("Session created with LLM backend: %s", backend_choice)

        # Seed conversation so the orchestrator knows about attachments
        if pdf_paths or custom_algo_path:
            pdf_names = [os.path.basename(p) for p in pdf_paths] if pdf_paths else []
            algo_name = os.path.basename(custom_algo_path) if custom_algo_path else None
            parts = []
            if pdf_names:
                parts.append(f"I have uploaded PDF papers: {', '.join(pdf_names)}.")
            if algo_name:
                parts.append(f"I have also uploaded my custom algorithm implementation: {algo_name}.")
            parts.append("I'll describe my problem now.")
            messages.append({"role": "user", "content": " ".join(parts)})

            reply_parts = []
            if pdf_names:
                reply_parts.append(
                    "I see your papers. I'll extract the algorithm specifications from them during intake "
                    "so you won't need to specify algorithms separately."
                )
            if algo_name:
                reply_parts.append(
                    f"Your custom algorithm ({algo_name}) is loaded and will be included in the benchmark automatically."
                )
            reply_parts.append("Go ahead — describe your optimization problem.")
            messages.append({"role": "assistant", "content": " ".join(reply_parts)})

        session_id = str(uuid.uuid4())
        _SESSIONS[session_id] = {"agent": agent, "messages": messages, "temp_dir": temp_dir}

    # Save user message to DB
    save_message(session_id, "user", request.message)

    # ── Launch orchestrator in a background thread ─────────────────────────
    # The thread always saves the result to DB when it finishes,
    # regardless of whether the SSE client is still connected.

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    _tools_called: list[str] = []
    _tool_records: list[dict] = []  # full tool call records for DB persistence
    _ui_metadata: dict = {}  # algorithmChoices, choicePrompt, etc.

    _last_tool_input: dict = {}

    def progress_cb(event: dict):
        """Called from the worker thread — enqueue event to the async queue."""
        if event.get("type") == "tool_start":
            _tools_called.append(event.get("tool", ""))
            _last_tool_input.clear()
            _last_tool_input.update(event.get("input", {}))
            _tool_records.append({"tool": event.get("tool", ""), "status": "running"})
        elif event.get("type") == "tool_end":
            # Mark the matching tool record as completed
            for t in reversed(_tool_records):
                if t["tool"] == event.get("tool") and t["status"] == "running":
                    t["status"] = "completed"
                    t["result"] = event.get("result")
                    break
        try:
            loop.call_soon_threadsafe(q.put_nowait, event)
        except RuntimeError:
            pass  # loop closed — SSE client gone, that's fine

        # Capture choice_prompt events for DB persistence
        if event.get("type") == "choice_prompt":
            _ui_metadata["choicePrompt"] = {
                "id": event.get("id"),
                "title": event.get("title"),
                "options": event.get("options"),
                "multiSelect": event.get("multi_select", False),
            }

        # After intake completes, emit structured algorithm choices for the UI
        if (
            event.get("type") == "tool_end"
            and event.get("tool") == "run_intake"
            and agent is not None
            and agent.state.algo_specs
            and agent.state.preferred_algo_spec_indices is None
        ):
            algos = [
                {
                    "index": i,
                    "name": s.name,
                    "approach": s.approach,
                    "source": s.source,
                }
                for i, s in enumerate(agent.state.algo_specs)
            ]
            _ui_metadata["algorithmChoices"] = algos
            try:
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "algorithm_select",
                    "algorithms": algos,
                })
            except RuntimeError:
                pass

        # After code_algorithm completes, emit algorithm_coded events (AI-coded only)
        if (
            event.get("type") == "tool_end"
            and event.get("tool") == "code_algorithm"
            and agent is not None
            and agent.state.algorithms
        ):
            custom_name = agent.state.custom_algo_name
            for algo_wrapper in agent.state.algorithms:
                name = getattr(algo_wrapper, "name", None)
                if name and name != custom_name:
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, {
                            "type": "algorithm_coded",
                            "name": name,
                        })
                    except RuntimeError:
                        pass

        # After load_suite with list_suites=true, emit suite choices
        if (
            event.get("type") == "tool_end"
            and event.get("tool") == "load_suite"
            and agent is not None
        ):
            result_text = event.get("result", "")
            if "Available benchmark suites" in result_text:
                # Parse suite info from the tool result
                import re as _re
                suite_opts = []
                for m in _re.finditer(
                    r"•\s+(\w+):\s+(.+?)\((\d+)\s+instances?\)\s*\n\s+(.+)",
                    result_text,
                ):
                    suite_opts.append({
                        "value": m.group(1),
                        "label": m.group(2).strip(),
                        "description": f"{m.group(3)} instances — {m.group(4).strip()}",
                    })
                if suite_opts:
                    _ui_metadata["choicePrompt"] = {
                        "id": "benchmark_suite",
                        "title": "Select a benchmark suite",
                        "options": suite_opts,
                        "multiSelect": False,
                    }
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, {
                            "type": "choice_prompt",
                            "id": "benchmark_suite",
                            "title": "Select a benchmark suite",
                            "options": suite_opts,
                            "multi_select": False,
                        })
                    except RuntimeError:
                        pass
            elif "Instances in" in result_text:
                # Parse instance list from the tool result
                import re as _re
                inst_opts = []
                for m in _re.finditer(
                    r"•\s+([\w._-]+)\s+\((\d+)\s+nodes?\)",
                    result_text,
                ):
                    inst_opts.append({
                        "value": m.group(1),
                        "label": m.group(1),
                        "description": f"{m.group(2)} nodes",
                    })
                if inst_opts:
                    _ui_metadata["choicePrompt"] = {
                        "id": "benchmark_instances",
                        "title": "Select instances to load",
                        "options": inst_opts,
                        "multiSelect": True,
                    }
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, {
                            "type": "choice_prompt",
                            "id": "benchmark_instances",
                            "title": "Select instances to load",
                            "options": inst_opts,
                            "multi_select": True,
                        })
                    except RuntimeError:
                        pass

    # Emit session_start immediately so the frontend can persist the session ID
    q.put_nowait({"type": "session_start", "session_id": session_id})

    # Mark session as actively processing
    _SESSIONS[session_id]["processing"] = True
    _SESSIONS[session_id]["event_queue"] = q

    _agent = agent
    _messages = messages
    _sid = session_id
    _user_msg = request.message

    def _run_orchestrator():
        """Blocking orchestrator work — runs in a daemon thread."""
        try:
            msgs_out, reply = _agent.run_one_turn(
                _messages, _user_msg, progress_cb=progress_cb
            )
            _SESSIONS[_sid]["messages"] = msgs_out

            # Handle plot image
            plot_image = None
            if "analyze_results" in _tools_called and _agent.state.last_plot_path:
                plot_path = _agent.state.last_plot_path
                if os.path.exists(plot_path):
                    import base64 as _b64
                    with open(plot_path, "rb") as f:
                        encoded = _b64.b64encode(f.read()).decode("ascii")
                    plot_image = f"data:image/png;base64,{encoded}"
                _agent.state.last_plot_path = None

            # Emit interactive choice prompts based on pipeline state
            # Only show instance source if code_algorithm was called this turn
            # AND user didn't already specify an instance source preference
            if (
                "code_algorithm" in _tools_called
                and _agent.state.instance_source is None
                and _agent.state.preferred_instance_source is None
                and _agent.state.algorithms
                and len(_agent.state.algorithms) > 0
            ):
                gens = (
                    _agent.state.config.instance_config.generators
                    if _agent.state.config else []
                )
                gen_desc = ""
                if gens:
                    gen_names = [g.type for g in gens]
                    gen_desc = f" ({', '.join(gen_names)})"
                instance_source_prompt = {
                    "id": "instance_source",
                    "title": "How would you like to provide instances?",
                    "options": [
                        {
                            "value": "generators",
                            "label": "Generators",
                            "description": f"Use proposed graph generators{gen_desc}",
                        },
                        {
                            "value": "custom",
                            "label": "Custom JSON",
                            "description": "Load from your own instance file",
                        },
                        {
                            "value": "benchmark suite",
                            "label": "Benchmark Suite",
                            "description": "Use standard benchmark instances (DIMACS, BiqMac, etc.)",
                        },
                    ],
                    "multiSelect": False,
                }
                _ui_metadata["choicePrompt"] = instance_source_prompt
                try:
                    loop.call_soon_threadsafe(q.put_nowait, {
                        "type": "choice_prompt",
                        **instance_source_prompt,
                        "multi_select": False,
                    })
                except RuntimeError:
                    pass

            # Always persist to DB (even if the SSE client disconnected)
            save_message(
                session_id=_sid, role="assistant",
                content=reply,
                tools=_tool_records if _tool_records else None,
                plot_image=plot_image,
                metadata=_ui_metadata if _ui_metadata else None,
            )

            # Notify queue — the SSE generator will pick this up if connected
            try:
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "done", "session_id": _sid,
                    "reply": reply, "plot_image": plot_image,
                })
            except RuntimeError:
                pass

        except Exception as e:
            save_message(session_id=_sid, role="assistant", content=f"Error: {e}")
            logger.exception("Orchestrator error: %s", e)
            try:
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "error", "session_id": _sid, "error": str(e),
                })
            except RuntimeError:
                pass
        finally:
            _SESSIONS[_sid]["processing"] = False

    thread = threading.Thread(target=_run_orchestrator, daemon=True)
    thread.start()

    # ── SSE stream — just reads from the queue ────────────────────────────
    async def event_stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=2.0)
                    yield f"data: {_safe_json(event)}\n\n"
                    # Terminal events — stop streaming
                    if event.get("type") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    if not _SESSIONS.get(_sid, {}).get("processing", False):
                        # Orchestrator finished while we weren't reading — drain
                        while not q.empty():
                            evt = q.get_nowait()
                            yield f"data: {_safe_json(evt)}\n\n"
                        break
                    # Keep-alive heartbeat so proxies don't kill the connection
                    yield f"data: {_safe_json({'type': 'heartbeat'})}\n\n"
        except Exception as e:
            logger.exception("Stream error: %s", e)
            yield f"data: {_safe_json({'type': 'error', 'session_id': _sid, 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
