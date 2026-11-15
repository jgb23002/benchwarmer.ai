"""Exercise the restored web turn without contacting any model provider."""

import copy
from pathlib import Path
from unittest.mock import patch

from benchwarmer.agents.backends import LLMResponse, TextBlock, ToolUseBlock
from benchwarmer.agents.intake import IntakeAgent
from benchwarmer.agents.orchestrator import OrchestratorAgent, PipelineState
from benchwarmer.config import AlgorithmSpec, IntakeResult


class StubBackend:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return next(self.responses)


def make_agent(backend):
    # Avoid constructing real SDK clients or reading API credentials.
    agent = OrchestratorAgent.__new__(OrchestratorAgent)
    agent.backend = backend
    agent.state = PipelineState()
    agent.intake_backend_name = "nemotron"
    agent._progress_cb = None
    return agent


def test_web_turn_streams_tools_and_keeps_updated_conversation_state():
    backend = StubBackend(
        LLMResponse(content=[ToolUseBlock(id="tool-1", name="configure", input={})], stop_reason="tool_use"),
        LLMResponse(content=[TextBlock(text="Ready to benchmark")], stop_reason="end_turn"),
    )
    agent = make_agent(backend)

    def configure(name, data):
        assert name == "configure"
        agent.state.config = agent._build_fallback_intake_config("maximum cut")
        return "Configuration ready"

    agent._dispatch_tool = configure
    progress = []
    messages, reply = agent.run_one_turn([], "benchmark a graph", progress_cb=progress.append)
    assert reply == "Ready to benchmark"
    assert [event["type"] for event in progress] == ["thinking", "tool_start", "tool_end", "thinking"]
    assert messages[0] == {"role": "user", "content": "benchmark a graph"}
    assert messages[2]["content"][0]["tool_use_id"] == "tool-1"
    assert "maximum_cut" in backend.requests[1]["system"]


def test_intake_preserves_paper_algorithms_and_uploaded_pdf_paths(tmp_path):
    agent = make_agent(StubBackend())
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 test")
    agent.state.pdf_paths = [str(paper)]
    config = agent._build_fallback_intake_config("maximum cut")
    spec = AlgorithmSpec(name="paper_cut", approach="Partition a graph", source="paper.pdf")
    result = IntakeResult(config=config, algorithms=[spec])
    with patch.object(IntakeAgent, "run", return_value=result) as intake, patch(
        "benchwarmer.agents.orchestrator._extract_user_preferences", return_value={}
    ):
        response = agent._tool_run_intake({"description": "maximum cut"})
    assert intake.call_args.kwargs["pdf_paths"] == [str(paper)]
    assert agent.state.algo_specs == [spec]
    assert "paper_cut" in response


def test_intake_retains_the_deterministic_fallback_when_a_provider_fails():
    agent = make_agent(StubBackend())
    with patch.object(IntakeAgent, "run", side_effect=RuntimeError("provider unavailable")):
        reply = agent._tool_run_intake({"description": "maximum cut on random graphs"})
    assert agent.state.config.problem_class == "maximum_cut"
    assert "fallback configuration" in reply
    assert agent.state.algo_specs == []


def test_research_pdf_can_be_read_with_the_installed_requirements():
    paper = Path(__file__).resolve().parents[1] / "Lec_10_Max_Cut.pdf"
    text = IntakeAgent._extract_pdf_text(str(paper))
    assert "cut" in text.lower()
