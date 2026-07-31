from __future__ import annotations

import asyncio

import claude_code_council as council


def _model(name: str, provider: str = "openai") -> dict[str, str]:
    return {"id": name.lower(), "name": name, "provider": provider}


def test_safe_error_keeps_known_local_guidance_but_hides_provider_details() -> None:
    assert council._safe_error(RuntimeError("OPENAI_API_KEY not set")) == "OPENAI_API_KEY not set"

    private_detail = "provider rejected request with private response detail"
    label = council._safe_error(RuntimeError(private_detail))

    assert label == "RuntimeError"
    assert private_detail not in label


def test_gemini_uses_header_not_query_parameter(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    def fake_post(url: str, **kwargs):
        observed["url"] = url
        observed.update(kwargs)
        return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "placeholder-key")
    monkeypatch.setattr(council._http, "post", fake_post)

    assert council._call_gemini("gemini-2.5-pro", "hello") == "ok"
    assert observed["headers"] == {"x-goog-api-key": "placeholder-key"}
    assert "params" not in observed
    assert "placeholder-key" not in str(observed["url"])


def test_peer_review_retains_each_rotated_label_map(monkeypatch) -> None:
    models = [_model("Alpha"), _model("Beta", "gemini"), _model("Gamma", "xai")]
    stage1 = [
        {"model": model, "response": f"answer-{model['name']}"}
        for model in models
    ]
    monkeypatch.setattr(council, "call_model", lambda model, prompt: "FINAL RANKING:\n1. Response A")

    reviews = asyncio.run(council.stage2_peer_review("question", stage1))

    assert reviews[0]["label_map"] == {
        "Response A": "Alpha",
        "Response B": "Beta",
        "Response C": "Gamma",
    }
    assert reviews[1]["label_map"] == {
        "Response A": "Beta",
        "Response B": "Gamma",
        "Response C": "Alpha",
    }
    assert reviews[2]["label_map"] == {
        "Response A": "Gamma",
        "Response B": "Alpha",
        "Response C": "Beta",
    }


def test_chairman_receives_deanonymization_map(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_call(model: dict, prompt: str) -> str:
        captured["prompt"] = prompt
        return "synthesis"

    monkeypatch.setattr(council, "call_model", fake_call)
    stage1 = [{"model": _model("Alpha"), "response": "answer"}]
    stage2 = [{
        "model": _model("Reviewer"),
        "review": "1. Response A",
        "label_map": {"Response A": "Alpha"},
    }]

    assert council.stage3_chairman("question", stage1, stage2) == "synthesis"
    assert "Label map: Response A=Alpha" in captured["prompt"]
