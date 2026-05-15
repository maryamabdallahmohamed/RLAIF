from src.agents.critic.cache import read, write, _cache_key


def test_write_and_read_roundtrip(tmp_path):
    write(tmp_path, "helpfulness", "my prompt", "my response", "perturbed text", 2)
    result = read(tmp_path, "helpfulness", "my prompt", "my response")
    assert result == ("perturbed text", 2)


def test_cache_miss_returns_none(tmp_path):
    result = read(tmp_path, "helpfulness", "prompt", "response")
    assert result is None


def test_different_constitution_names_do_not_collide(tmp_path):
    write(tmp_path, "helpfulness", "prompt", "response", "perturbed A", 0)
    write(tmp_path, "truthfulness", "prompt", "response", "perturbed B", 1)

    assert read(tmp_path, "helpfulness", "prompt", "response") == ("perturbed A", 0)
    assert read(tmp_path, "truthfulness", "prompt", "response") == ("perturbed B", 1)


def test_different_prompts_do_not_collide(tmp_path):
    write(tmp_path, "helpfulness", "prompt A", "response", "perturbed A", 0)
    write(tmp_path, "helpfulness", "prompt B", "response", "perturbed B", 1)

    assert read(tmp_path, "helpfulness", "prompt A", "response") == ("perturbed A", 0)
    assert read(tmp_path, "helpfulness", "prompt B", "response") == ("perturbed B", 1)


def test_cache_key_is_deterministic():
    k1 = _cache_key("helpfulness", "p", "r")
    k2 = _cache_key("helpfulness", "p", "r")
    assert k1 == k2


def test_write_creates_cache_dir_if_missing(tmp_path):
    nested = tmp_path / "a" / "b" / "cache"
    write(nested, "helpfulness", "p", "r", "perturbed", 0)
    assert nested.exists()
