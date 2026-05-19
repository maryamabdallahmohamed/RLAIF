import json

import pytest
import yaml

from src.agents.critic.agent import CriticAgent, CriticParseError


@pytest.fixture
def constitutions_file(tmp_path):
    data = {
        "helpfulness": {
            "principle": "helpfulness",
            "system_prompt": "Modify one sentence to be less helpful.",
        }
    }
    p = tmp_path / "constitutions.yaml"
    p.write_text(yaml.dump(data))
    return p


def test_perturb_returns_parsed_response(constitutions_file, tmp_path):
    fake_reply = json.dumps({"sentence_idx": 1, "perturbed_response": "Modified response."})

    def fake_client(url, model, system_prompt, user_message):
        return fake_reply

    critic = CriticAgent(
        ollama_url="http://localhost:11434",
        model="test-model",
        constitutions_path=constitutions_file,
        cache_dir=tmp_path / "cache",
        _client_fn=fake_client,
    )

    perturbed, idx = critic.perturb("What is 2+2?", "2+2 equals 4.", "helpfulness")
    assert perturbed == "Modified response."
    assert idx == 1


def test_perturb_malformed_json_raises_critic_parse_error(constitutions_file, tmp_path):
    def fake_client(url, model, system_prompt, user_message):
        return "not valid json at all"

    critic = CriticAgent(
        ollama_url="http://localhost:11434",
        model="test-model",
        constitutions_path=constitutions_file,
        cache_dir=tmp_path / "cache",
        _client_fn=fake_client,
    )

    with pytest.raises(CriticParseError):
        critic.perturb("What is 2+2?", "2+2 equals 4.", "helpfulness")


def test_perturb_cache_hit_skips_client(constitutions_file, tmp_path):
    call_count = {"n": 0}

    def fake_client(url, model, system_prompt, user_message):
        call_count["n"] += 1
        return json.dumps({"sentence_idx": 0, "perturbed_response": "First call result."})

    critic = CriticAgent(
        ollama_url="http://localhost:11434",
        model="test-model",
        constitutions_path=constitutions_file,
        cache_dir=tmp_path / "cache",
        _client_fn=fake_client,
    )

    critic.perturb("prompt", "response", "helpfulness")
    critic.perturb("prompt", "response", "helpfulness")  # identical inputs — must hit cache

    assert call_count["n"] == 1


def test_perturb_markdown_wrapped_json_is_parsed(constitutions_file, tmp_path):
    # Some models wrap JSON in ```json ... ``` even when told not to
    inner = json.dumps({"sentence_idx": 2, "perturbed_response": "Wrapped response."})
    fake_reply = f"```json\n{inner}\n```"

    def fake_client(url, model, system_prompt, user_message):
        return fake_reply

    critic = CriticAgent(
        ollama_url="http://localhost:11434",
        model="test-model",
        constitutions_path=constitutions_file,
        cache_dir=tmp_path / "cache",
        _client_fn=fake_client,
    )

    perturbed, idx = critic.perturb("p", "r", "helpfulness")
    assert perturbed == "Wrapped response."
    assert idx == 2


def test_perturb_unknown_constitution_raises_critic_parse_error(constitutions_file, tmp_path):
    def fake_client(url, model, system_prompt, user_message):
        return "{}"

    critic = CriticAgent(
        ollama_url="http://localhost:11434",
        model="test-model",
        constitutions_path=constitutions_file,
        cache_dir=tmp_path / "cache",
        _client_fn=fake_client,
    )

    with pytest.raises(CriticParseError):
        critic.perturb("p", "r", "nonexistent_constitution")
