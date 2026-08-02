"""CLI construction respects provider policy before clients are created."""

from __future__ import annotations

import src.main as cli


def test_provider_kill_switch_blocks_clients_and_is_reflected_in_config(
    monkeypatch,
):
    captured = []
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setenv("CADENCE_DISABLE_PROVIDER", "1")
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "build_companion",
        lambda config, deps: captured.append((config, deps)) or object(),
    )

    cli._build_companion()
    config, deps = captured[-1]
    assert not config.use_live_embedder and not config.use_generator
    assert deps.live_embedder is None and deps.generator is None


def test_cli_local_only_policy_avoids_even_constructing_provider_helpers(monkeypatch):
    captured = []
    monkeypatch.delenv("CADENCE_DISABLE_PROVIDER", raising=False)
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "_live_embedder",
        lambda: (_ for _ in ()).throw(AssertionError("embedder constructed")),
    )
    monkeypatch.setattr(
        cli,
        "_text_generator",
        lambda: (_ for _ in ()).throw(AssertionError("generator constructed")),
    )
    monkeypatch.setattr(
        cli,
        "build_companion",
        lambda config, deps: captured.append((config, deps)) or object(),
    )

    cli._build_companion(provider_enabled=False)
    config, deps = captured[-1]
    assert not config.use_live_embedder and not config.use_generator
    assert deps.live_embedder is None and deps.generator is None
