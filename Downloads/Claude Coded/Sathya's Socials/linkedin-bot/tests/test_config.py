import json
import os
import pytest
from unittest.mock import patch, mock_open, MagicMock


def test_load_context_both_empty(tmp_path):
    """Returns empty strings and two warnings when both files are empty."""
    posts = tmp_path / "my_posts.txt"
    icp = tmp_path / "icp.txt"
    posts.write_text("")
    icp.write_text("")

    from config import load_context
    ctx = load_context(posts_path=str(posts), icp_path=str(icp))

    assert ctx["posts_text"] == ""
    assert ctx["icp_text"] == ""
    assert len(ctx["warnings"]) == 2


def test_load_context_with_content(tmp_path):
    """Returns content and no warnings when both files have text."""
    posts = tmp_path / "my_posts.txt"
    icp = tmp_path / "icp.txt"
    posts.write_text("Sample post content")
    icp.write_text("Target: founders")

    from config import load_context
    ctx = load_context(posts_path=str(posts), icp_path=str(icp))

    assert ctx["posts_text"] == "Sample post content"
    assert ctx["icp_text"] == "Target: founders"
    assert ctx["warnings"] == []


def test_get_account_id_from_cache(tmp_path):
    """Returns accountId from config.json without calling the API."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"linkedin_account_id": "abc123"}))

    from config import get_account_id
    account_id = get_account_id(
        late_api_key="fake_key",
        config_path=str(config_file)
    )

    assert account_id == "abc123"


def test_get_account_id_fetches_and_caches(tmp_path):
    """Fetches from GetLate API on first run and saves to config.json."""
    config_file = tmp_path / "config.json"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"id": "xyz789", "platform": "linkedin", "name": "Test User"},
        {"id": "other1", "platform": "twitter", "name": "Test User"},
    ]

    with patch("requests.get", return_value=mock_response):
        from config import get_account_id
        account_id = get_account_id(
            late_api_key="fake_key",
            config_path=str(config_file)
        )

    assert account_id == "xyz789"
    saved = json.loads(config_file.read_text())
    assert saved["linkedin_account_id"] == "xyz789"


def test_get_account_id_no_linkedin_account(tmp_path):
    """Raises SystemExit with helpful message when no LinkedIn account found."""
    config_file = tmp_path / "config.json"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"id": "other1", "platform": "twitter", "name": "Test User"},
    ]

    with patch("requests.get", return_value=mock_response):
        from config import get_account_id
        with pytest.raises(SystemExit):
            get_account_id(late_api_key="fake_key", config_path=str(config_file))


def test_get_env_missing_keys(monkeypatch):
    """Raises SystemExit when API keys are missing from environment."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LATE_API_KEY", raising=False)
    from config import get_env
    with pytest.raises(SystemExit):
        get_env()
