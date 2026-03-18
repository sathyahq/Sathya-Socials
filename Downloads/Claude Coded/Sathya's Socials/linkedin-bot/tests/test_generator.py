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
    assert "Start with this exact line:" in user_msg
    assert "Winning hook line." in user_msg


def test_format_post_runs_all_three_stages():
    """format_post() runs pre-pass, LLM, and post-pass in sequence."""
    formatted = "Hook line.\nRehook line.\n\nBody block.\n\nCTA here."
    client = make_mock_client(formatted)

    from generator import format_post
    original = "Hook line.\n\nRehook line.\n\nBody block.\n\nCTA here."
    result = format_post(client, original)

    assert result == formatted
    assert "—" not in result


def test_format_post_strips_em_dash_before_llm():
    """format_post() removes em-dashes before sending to LLM."""
    client = make_mock_client("Hook line, very good.\n\nBody block.")

    from generator import format_post
    result = format_post(client, "Hook line — very good.\n\nBody block.")
    assert "—" not in result


def test_generate_calls_format_post_as_pass_4(capsys):
    """generate() calls format_post and prints 'Formatting post...'."""
    hooks_text = "1. Hook one\n2. Hook two\n3. Hook three"
    score_text = "2"
    post_text = "Hook two\n\nThis is the full post content."
    formatted_text = "Hook two\nThis is the full post content."

    call_count = {"n": 0}
    responses = [hooks_text, score_text, post_text, formatted_text]

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

    assert post == formatted_text
    assert char_count == len(formatted_text)
    captured = capsys.readouterr()
    assert "Formatting post" in captured.out
    assert mock_client.chat.completions.create.call_count == 4


def test_llm_format_returns_formatted_text():
    """_llm_format sends post to Groq and returns reformatted text."""
    formatted = "Hook line.\nRehook line.\n\nBody block.\n\nCTA here."
    client = make_mock_client(formatted)

    from generator import _llm_format
    result = _llm_format(client, "Hook line.\n\nRehook line.\n\nBody block.\n\nCTA here.")

    assert result == formatted
    assert client.chat.completions.create.called


def test_llm_format_prompt_instructs_no_word_changes():
    """_llm_format prompt explicitly tells LLM not to change words."""
    client = make_mock_client("Formatted post.")

    from generator import _llm_format
    _llm_format(client, "Some post text.")

    call_args = client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][-1]["content"]
    assert "do not change" in user_msg.lower() or "Do NOT change" in user_msg


def test_llm_format_prompt_includes_grouping_rules():
    """_llm_format prompt contains the paragraph grouping rules."""
    client = make_mock_client("Formatted post.")

    from generator import _llm_format
    _llm_format(client, "Some post text.")

    call_args = client.chat.completions.create.call_args
    system_msg = call_args[1]["messages"][0]["content"]
    assert "→" in system_msg
    assert "blank line" in system_msg.lower()


def test_post_pass_passes_clean_output():
    """_post_pass returns text unchanged when formatting is correct."""
    from generator import _post_pass
    original = "Clean post.\n\nSecond block."
    formatted = "Clean post.\n\nSecond block."
    result = _post_pass(formatted, original)
    assert result == formatted


def test_post_pass_raises_on_em_dash():
    """_post_pass raises ValueError if em-dash survived formatting."""
    from generator import _post_pass
    original = "Some post text here."
    formatted = "Some post — with em-dash."
    with pytest.raises(ValueError, match="em-dash"):
        _post_pass(formatted, original)


def test_post_pass_raises_on_excessive_length_change():
    """_post_pass raises ValueError if formatted post grew more than 15%."""
    from generator import _post_pass
    original = "A" * 1000
    formatted = "A" * 1200  # 20% longer — over the 15% threshold
    with pytest.raises(ValueError, match="length drift"):
        _post_pass(formatted, original)


def test_post_pass_allows_length_shrink_within_threshold():
    """_post_pass allows up to 15% length change in either direction."""
    from generator import _post_pass
    original = "A" * 1000
    formatted = "A" * 880  # 12% shorter — within threshold
    result = _post_pass(formatted, original)
    assert result == formatted


def test_pre_pass_strips_em_dashes():
    """_pre_pass replaces em-dashes with commas."""
    from generator import _pre_pass
    result = _pre_pass("This is great — really great — trust me.")
    assert "—" not in result
    assert "," in result


def test_pre_pass_collapses_double_spaces():
    """_pre_pass collapses double spaces to single spaces."""
    from generator import _pre_pass
    result = _pre_pass("Hello  world  here.")
    assert "  " not in result


def test_pre_pass_leaves_clean_text_unchanged():
    """_pre_pass does not modify text that needs no fixes."""
    from generator import _pre_pass
    clean = "Clean text here.\n\nSecond paragraph."
    assert _pre_pass(clean) == clean



def test_generate_returns_post_and_char_count():
    """generate() orchestrates all 4 passes and returns (post_text, char_count)."""
    hooks_text = "1. Hook one\n2. Hook two\n3. Hook three"
    score_text = "2"
    post_text = "Hook two\n\nThis is the full post content."
    formatted_text = "Hook two\n\nThis is the full post content."

    call_count = {"n": 0}
    responses = [hooks_text, score_text, post_text, formatted_text]

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

    assert post == formatted_text
    assert char_count == len(formatted_text)
