from src.agents.critic.segment import segment


def test_empty_string_returns_empty_list():
    assert segment("") == []


def test_single_sentence():
    result = segment("Hello world.")
    assert result == ["Hello world."]


def test_multiple_sentences():
    result = segment("First sentence. Second sentence. Third sentence.")
    assert len(result) == 3
    assert result[0] == "First sentence."


def test_fenced_code_block_is_atomic():
    text = "Here is some code.\n```python\nx = 1\ny = 2\n```\nEnd of example."
    result = segment(text)
    code_blocks = [s for s in result if "```" in s]
    assert len(code_blocks) == 1
    assert "x = 1" in code_blocks[0]


def test_display_math_block_is_atomic():
    text = "Consider the equation. $$E = mc^2$$ This is important."
    result = segment(text)
    math_blocks = [s for s in result if "$$" in s]
    assert len(math_blocks) == 1
    assert "E = mc^2" in math_blocks[0]
