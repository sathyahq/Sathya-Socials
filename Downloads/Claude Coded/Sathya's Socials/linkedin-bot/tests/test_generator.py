import pytest
from unittest.mock import MagicMock, patch


def make_mock_client(response_text):
    """Helper: returns a mock OpenAI client that yields response_text."""
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


def test_build_system_prompt_with_context():
    """System prompt includes rules, voice samples, and ICP when provided."""
    from generator import build_system_prompt
    prompt = build_system_prompt(
        posts_text="My sample post here",
        icp_text="Founders aged 25-40"
    )
    assert "LINKEDIN" in prompt
    assert "My sample post here" in prompt
    assert "Founders aged 25-40" in prompt


def test_build_system_prompt_without_context():
    """System prompt excludes voice/ICP sections when files are empty."""
    from generator import build_system_prompt
    prompt = build_system_prompt(posts_text="", icp_text="")
    assert "LINKEDIN" in prompt
    assert "VOICE SAMPLES" not in prompt
    assert "TARGET AUDIENCE" not in prompt


def test_generate_hooks_returns_list():
    """generate_hooks parses numbered list from Groq and returns list of strings."""
    hooks_text = "1. Hook one here\n2. Hook two here\n3. Hook three here"
    client = make_mock_client(hooks_text)

    from generator import generate_hooks
    hooks = generate_hooks(client, topic="AI mistakes", system_prompt="rules")

    assert isinstance(hooks, list)
    assert len(hooks) == 3
    assert hooks[0] == "Hook one here"


def test_score_hooks_returns_valid_index():
    """score_hooks returns an integer index within the hooks list."""
    hooks = ["Hook A", "Hook B", "Hook C"]
    score_response = "2\nReason: Hook B has strong dwell time potential."
    client = make_mock_client(score_response)

    from generator import score_hooks
    idx = score_hooks(client, hooks=hooks, system_prompt="rules")

    assert isinstance(idx, int)
    assert 0 <= idx < len(hooks)


def test_score_hooks_falls_back_to_zero_on_bad_response():
    """score_hooks returns 0 if Groq returns unparseable output."""
    hooks = ["Hook A", "Hook B"]
    client = make_mock_client("I think the best one is Hook A because...")

    from generator import score_hooks
    idx = score_hooks(client, hooks=hooks, system_prompt="rules")

    assert idx == 0


def test_score_hooks_falls_back_to_zero_on_out_of_range():
    """score_hooks returns 0 if Groq returns a number beyond the hooks list."""
    hooks = ["Hook A", "Hook B"]
    client = make_mock_client("7\nHook 7 is best because it is very impactful.")

    from generator import score_hooks
    idx = score_hooks(client, hooks=hooks, system_prompt="rules")

    assert idx == 0


def test_generate_post_uses_winning_hook():
    """generate_post sends winning hook to Groq and returns post text."""
    post_text = "Winning hook line.\n\nBody of the post here.\n\nWhat would you add?"
    client = make_mock_client(post_text)

    from generator import generate_post
    result = generate_post(
        client,
        topic="AI mistakes",
        winning_hook="Winning hook line.",
        system_prompt="rules"
    )

    assert result == post_text
    call_args = client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][-1]["content"]
    assert "Use EXACTLY this as your first line" in user_msg
    assert "Winning hook line." in user_msg


def test_generate_returns_post_and_char_count():
    """generate() orchestrates all 3 passes and returns (post_text, char_count)."""
    hooks_text = "1. Hook one\n2. Hook two\n3. Hook three"
    score_text = "2"
    post_text = "Hook two\n\nThis is the full post content."

    call_count = {"n": 0}
    responses = [hooks_text, score_text, post_text]

    def side_effect(**kwargs):
        mock_choice = MagicMock()
        mock_choice.message.content = responses[call_count["n"]]
        call_count["n"] += 1
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        return mock_completion

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect

    with patch("generator.OpenAI", return_value=mock_client):
        from generator import generate
        post, char_count = generate(
            topic="AI mistakes",
            groq_key="fake",
            posts_text="",
            icp_text=""
        )

    assert post == post_text
    assert char_count == len(post_text)
