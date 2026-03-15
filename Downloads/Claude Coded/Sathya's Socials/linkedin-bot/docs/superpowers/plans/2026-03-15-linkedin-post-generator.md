# LinkedIn Post Generator Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool that takes a topic, generates an optimised LinkedIn post via 3-pass Groq, auto-selects the best hook, shows it for approval, then schedules it via GetLate at the next optimal IST time slot.

**Architecture:** Three Groq passes (generate hooks → score hooks → generate post around winner) feed into a terminal approval flow. On approval, a slot-picker finds the next optimal IST window and POSTs to GetLate. Config and account ID are cached locally in `config.json`.

**Tech Stack:** Python 3.10+, `openai` SDK (Groq-compatible), `requests`, `python-dotenv`, `pytest`

---

## Chunk 1: Scaffold, Rules, Config

### Task 1: Project Scaffold

**Files:**
- Create: `linkedin-bot/requirements.txt`
- Create: `linkedin-bot/.env.example`
- Create: `linkedin-bot/.gitignore`
- Create: `linkedin-bot/my_posts.txt`
- Create: `linkedin-bot/icp.txt`
- Create: `linkedin-bot/tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
openai
requests
python-dotenv
pytest
```

- [ ] **Step 2: Create `.env.example`**

```
GROQ_API_KEY=your_groq_key_here
LATE_API_KEY=your_getlate_key_here
```

- [ ] **Step 3: Create `.gitignore`**

```
.env
config.json
pending_posts.json
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Create empty placeholder files**

`my_posts.txt` — empty file (user fills in later)
`icp.txt` — empty file (user fills in later)
`tests/__init__.py` — empty file

- [ ] **Step 5: Install dependencies**

```bash
cd linkedin-bot
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 6: Commit**

```bash
git add linkedin-bot/requirements.txt linkedin-bot/.env.example linkedin-bot/.gitignore linkedin-bot/my_posts.txt linkedin-bot/icp.txt linkedin-bot/tests/__init__.py
git commit -m "chore: scaffold linkedin-bot project structure"
```

---

### Task 2: `linkedin_rules.py` — Best Practices Constant

**Files:**
- Create: `linkedin-bot/linkedin_rules.py`

No tests needed — this is a pure data constant.

- [ ] **Step 1: Create `linkedin-bot/linkedin_rules.py`**

```python
LINKEDIN_RULES = """
## LINKEDIN ALGORITHM (2025)
- LinkedIn prioritizes DWELL TIME — posts people read fully beat posts that just get likes
- Golden window: first 60 minutes after posting are critical. Early engagement triggers wider distribution
- Topic authority matters: consistent posting on one niche = algorithm boosts your content more
- No engagement bait: "Comment YES if you agree" now gets penalized. Ask genuine questions instead
- External links suppress reach by up to 80%. Never put links in post body. Put them in first comment instead
- Posting more than once in 24 hours causes the newer post to cannibalize the previous one's reach
- More than 5 hashtags triggers a spam penalty. Use 0–3 max (hashtags have reduced impact in 2025)
- Avoid bit.ly or link shorteners — they trigger spam filters

## POST LENGTH
- Sweet spot: 1,300–1,600 characters for maximum engagement and discussion
- Under 300 characters: works for punchy opinions or questions only
- Above 2,000 characters: diminishing returns unless audience is deeply invested
- Sentences: max 12 words per sentence. Short = more readable = more dwell time
- Paragraphs: max 1–2 sentences each. White space is not wasted space

## HOOK (CRITICAL — make or break)
- First 210 characters are visible before "See More" cutoff — this determines if they click
- Hook must be under 8 words ideally. Bold, direct, surprising
- Best hook types: surprising stat, pain point, contrarian opinion, "How I" personal story, direct question
- "How I" beats "How to" — people want real stories, not generic advice
- Never open with: "I hope everyone is doing well", "Happy Monday", or any pleasantry
- Always include a second "rehook" on line 2 to build tension and pull reader in further

## FORMATTING
- Use line breaks between every paragraph (single line paragraphs)
- Use → • ✓ as visual cues instead of standard bullet points
- Mobile-first: 60%+ of LinkedIn users browse on phone
- No walls of text. Each line should be easy to skim
- No orphan words (single word on its own line at the end of a sentence)
- Emojis: 1–3 max, used sparingly and only where they add meaning

## POST FRAMEWORKS TO ROTATE
1. Story Hook → Problem → Insight → CTA
2. ABT: And (setup) → But (conflict) → Therefore (resolution)
3. List / Tips: Hook → numbered or bulleted value points → CTA
4. Contrarian Opinion: Challenge common belief → explain why → back with personal evidence → invite debate
5. "I believed X... but learned Y" — vulnerable, relatable, high saves

## CALLS TO ACTION (CTA)
- Always end with ONE clear CTA
- Examples: "Have you faced this? Drop it in the comments." / "Save this if you want to come back to it." / "What would you add?"
- Avoid: "Like and share if you agree" (engagement bait penalty)

## TIMING (IST — Asia/Kolkata)
Best slots in order:
1. Tuesday 8:00–9:00 AM IST
2. Wednesday 8:00–9:00 AM IST
3. Thursday 8:00–9:00 AM IST
4. Tuesday–Thursday 5:00–6:00 PM IST (secondary)
5. Monday/Friday: lower performance, use only if above slots are filled

## WHAT TO AVOID
- Generic AI phrases: "In today's fast-paced world", "leverage", "synergy", "game-changer", "dive into"
- Passive voice
- Starting with "I" (LinkedIn algorithm slightly penalizes this)
- Self-promotion without value
- Hashtag stuffing
- Corporate jargon
"""
```

- [ ] **Step 2: Commit**

```bash
git add linkedin-bot/linkedin_rules.py
git commit -m "feat: add LinkedIn best practices rules constant"
```

---

### Task 3: `config.py` — Env + Account ID Cache

**Files:**
- Create: `linkedin-bot/config.py`
- Create: `linkedin-bot/tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Create `linkedin-bot/tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd linkedin-bot
pytest tests/test_config.py -v
```

Expected: ImportError or ModuleNotFoundError (config.py doesn't exist yet).

- [ ] **Step 3: Create `linkedin-bot/config.py`**

```python
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DEFAULT_POSTS_PATH = os.path.join(BASE_DIR, "my_posts.txt")
DEFAULT_ICP_PATH = os.path.join(BASE_DIR, "icp.txt")

GETLATE_BASE = "https://getlate.dev/api/v1"


def load_context(posts_path=DEFAULT_POSTS_PATH, icp_path=DEFAULT_ICP_PATH):
    """Load voice samples and ICP from files. Return content + warnings."""
    warnings = []
    posts_text = ""
    icp_text = ""

    if os.path.exists(posts_path):
        posts_text = open(posts_path).read().strip()
    if not posts_text:
        warnings.append(
            "my_posts.txt is empty — generating without voice samples.\n"
            "    Add your LinkedIn posts to improve tone matching."
        )

    if os.path.exists(icp_path):
        icp_text = open(icp_path).read().strip()
    if not icp_text:
        warnings.append(
            "icp.txt is empty — generating without ICP context.\n"
            "    Add your target audience description to improve relevance."
        )

    return {"posts_text": posts_text, "icp_text": icp_text, "warnings": warnings}


def get_account_id(late_api_key, config_path=DEFAULT_CONFIG_PATH):
    """Return cached LinkedIn accountId, or fetch from GetLate and cache it."""
    if os.path.exists(config_path):
        data = json.loads(open(config_path).read())
        if data.get("linkedin_account_id"):
            return data["linkedin_account_id"]

    resp = requests.get(
        f"{GETLATE_BASE}/accounts",
        headers={"Authorization": f"Bearer {late_api_key}"},
    )
    resp.raise_for_status()
    accounts = resp.json()

    linkedin = next(
        (a for a in accounts if a.get("platform") == "linkedin"), None
    )
    if not linkedin:
        print(
            "\n❌ No LinkedIn account found in GetLate.\n"
            "   Connect your LinkedIn at: https://getlate.dev/dashboard/connections\n"
        )
        sys.exit(1)

    account_id = linkedin["id"]
    with open(config_path, "w") as f:
        json.dump({"linkedin_account_id": account_id}, f, indent=2)

    return account_id


def get_env():
    """Load and validate required environment variables."""
    groq_key = os.getenv("GROQ_API_KEY")
    late_key = os.getenv("LATE_API_KEY")

    missing = []
    if not groq_key:
        missing.append("GROQ_API_KEY")
    if not late_key:
        missing.append("LATE_API_KEY")

    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("   Copy .env.example to .env and fill in your API keys.\n")
        sys.exit(1)

    return {"groq_key": groq_key, "late_key": late_key}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd linkedin-bot
pytest tests/test_config.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add linkedin-bot/config.py linkedin-bot/tests/test_config.py
git commit -m "feat: add config loader with accountId caching and context loading"
```

---

## Chunk 2: Generator (3-Pass Groq)

### Task 4: `generator.py` — Hook Generation, Scoring, Post Writing

**Files:**
- Create: `linkedin-bot/generator.py`
- Create: `linkedin-bot/tests/test_generator.py`

- [ ] **Step 1: Write failing tests**

Create `linkedin-bot/tests/test_generator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd linkedin-bot
pytest tests/test_generator.py -v
```

Expected: ImportError (generator.py doesn't exist yet).

- [ ] **Step 3: Create `linkedin-bot/generator.py`**

```python
import re
import sys

from openai import OpenAI

from linkedin_rules import LINKEDIN_RULES

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"


def build_system_prompt(posts_text: str, icp_text: str) -> str:
    """Build the system prompt injected into every Groq call."""
    prompt = f"## LINKEDIN BEST PRACTICES\n{LINKEDIN_RULES}\n\n"

    if posts_text:
        prompt += f"## VOICE SAMPLES (write exactly like these)\n{posts_text}\n\n"

    if icp_text:
        prompt += f"## TARGET AUDIENCE\n{icp_text}\n\n"

    prompt += (
        "Write exactly like the voice samples. "
        "Follow every rule in the knowledge base. "
        "Do NOT use generic AI phrasing. "
        "Do NOT start with 'I'. "
        "Write for a human professional."
    )
    return prompt


def generate_hooks(client, topic: str, system_prompt: str) -> list[str]:
    """Pass 1: Generate 3–5 hook variations for the topic."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    "Generate 5 LinkedIn hook variations for this topic.\n"
                    "Rules:\n"
                    "- Each hook must be under 210 characters\n"
                    "- Each hook must be under 8 words ideally\n"
                    "- Do NOT start with 'I'\n"
                    "- Use varied hook types: stat, pain point, contrarian, 'How I', question\n"
                    "- No pleasantries\n\n"
                    "Output ONLY a numbered list, one hook per line. No explanations."
                ),
            },
        ],
    )
    raw = resp.choices[0].message.content.strip()
    hooks = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading "1. " "2. " etc.
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned:
            hooks.append(cleaned)
    return hooks


def score_hooks(client, hooks: list[str], system_prompt: str) -> int:
    """Pass 2: Score hooks and return index (0-based) of the best one."""
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hooks))
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Score these LinkedIn hooks and pick the best one:\n\n{numbered}\n\n"
                    "Evaluate each on:\n"
                    "- Dwell time potential (will people read on?)\n"
                    "- First 210 chars impact\n"
                    "- Hook type strength\n"
                    "- No banned phrases\n"
                    "- Does NOT start with 'I'\n\n"
                    "Reply with ONLY the number of the best hook on the first line, "
                    "then one sentence explaining why."
                ),
            },
        ],
    )
    raw = resp.choices[0].message.content.strip()
    first_line = raw.splitlines()[0].strip()
    match = re.search(r"\d+", first_line)
    if match:
        idx = int(match.group()) - 1
        if 0 <= idx < len(hooks):
            return idx
    return 0  # fallback to first hook


def generate_post(client, topic: str, winning_hook: str, system_prompt: str) -> str:
    """Pass 3: Generate full LinkedIn post using the winning hook as the opener."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Use EXACTLY this as your first line (do not change it):\n{winning_hook}\n\n"
                    "Now write the full LinkedIn post.\n"
                    "Requirements:\n"
                    "- 1,300–1,600 characters total\n"
                    "- Single-sentence paragraphs with line breaks between each\n"
                    "- Include a second 'rehook' on line 2 to build tension\n"
                    "- Max 12 words per sentence\n"
                    "- End with ONE clear CTA (no engagement bait)\n"
                    "- 0–3 hashtags max, placed at the end\n"
                    "- No external links in the post body\n"
                    "- No generic AI phrases\n\n"
                    "Output ONLY the post. No explanations, no preamble."
                ),
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def generate(topic: str, groq_key: str, posts_text: str, icp_text: str) -> tuple[str, int]:
    """Orchestrate all 3 passes. Return (post_text, char_count)."""
    client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)
    system_prompt = build_system_prompt(posts_text, icp_text)

    print("⚙️  Generating hooks...")
    hooks = generate_hooks(client, topic, system_prompt)

    print("⚙️  Scoring hooks...", end=" ", flush=True)
    best_idx = score_hooks(client, hooks, system_prompt)
    winning_hook = hooks[best_idx]
    print(f'selected: "Hook {best_idx + 1}"')

    print("⚙️  Writing full post...")
    post = generate_post(client, topic, winning_hook, system_prompt)

    return post, len(post)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd linkedin-bot
pytest tests/test_generator.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add linkedin-bot/generator.py linkedin-bot/tests/test_generator.py
git commit -m "feat: add 3-pass Groq generator with hook scoring"
```

---

## Chunk 3: Scheduler + Main + README

### Task 5: `scheduler.py` — Slot Picker + GetLate Integration

**Files:**
- Create: `linkedin-bot/scheduler.py`
- Create: `linkedin-bot/tests/test_scheduler.py`

- [ ] **Step 1: Write failing tests**

Create `linkedin-bot/tests/test_scheduler.py`:

```python
import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def make_ist(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def test_next_slot_from_monday_morning():
    """From Monday morning, next slot is Tuesday 8 AM."""
    # Monday 2026-03-16 09:00 IST
    now = make_ist(2026, 3, 16, 9, 0)
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now)
    assert slot.weekday() == 1  # Tuesday
    assert slot.hour == 8
    assert slot.minute == 0


def test_next_slot_from_tuesday_before_8am():
    """From Tuesday 7 AM, next slot is Tuesday 8 AM same day."""
    now = make_ist(2026, 3, 17, 7, 0)
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now)
    assert slot.weekday() == 1  # Tuesday
    assert slot.hour == 8
    assert slot.date() == now.date()


def test_next_slot_from_tuesday_after_8am_before_5pm():
    """From Tuesday 10 AM, next slot is Tuesday 5 PM."""
    now = make_ist(2026, 3, 17, 10, 0)
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now)
    assert slot.weekday() == 1  # Tuesday
    assert slot.hour == 17


def test_next_slot_from_tuesday_evening():
    """From Tuesday 6 PM, next slot is Wednesday 8 AM."""
    now = make_ist(2026, 3, 17, 18, 30)
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now)
    assert slot.weekday() == 2  # Wednesday
    assert slot.hour == 8


def test_next_slot_from_thursday_evening():
    """From Thursday 7 PM, next slot is Friday 8 AM (Mon/Fri fallback tier)."""
    now = make_ist(2026, 3, 19, 19, 0)
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now)
    assert slot.weekday() == 4  # Friday
    assert slot.hour == 8


def test_next_slot_skips_within_24h(tmp_path):
    """Skips a slot if a post is already scheduled within 24h of it."""
    pending_file = tmp_path / "pending_posts.json"
    # Already scheduled Tuesday 8 AM
    pending_file.write_text(json.dumps([
        {"scheduled_for": "2026-03-17T08:00:00+05:30"}
    ]))

    now = make_ist(2026, 3, 16, 9, 0)  # Monday 9 AM
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now, pending_path=str(pending_file))

    # Should skip Tuesday 8 AM and give Tuesday 5 PM or later
    assert not (slot.weekday() == 1 and slot.hour == 8)


def test_schedule_post_success(tmp_path):
    """Returns True and does not write to pending on success."""
    pending_file = tmp_path / "pending_posts.json"

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"id": "post123"}

    from datetime import timezone, timedelta
    slot = datetime(2026, 3, 17, 8, 0, tzinfo=IST)

    with patch("requests.post", return_value=mock_response):
        from scheduler import schedule_post
        success = schedule_post(
            content="Test post",
            account_id="acc123",
            late_api_key="fake_key",
            slot=slot,
            pending_path=str(pending_file)
        )

    assert success is True
    assert not pending_file.exists()


def test_schedule_post_failure_saves_pending(tmp_path):
    """Returns False and saves to pending_posts.json on API failure."""
    pending_file = tmp_path / "pending_posts.json"

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = Exception("Server error")

    slot = datetime(2026, 3, 17, 8, 0, tzinfo=IST)

    with patch("requests.post", side_effect=Exception("Connection error")):
        from scheduler import schedule_post
        success = schedule_post(
            content="Test post",
            account_id="acc123",
            late_api_key="fake_key",
            slot=slot,
            pending_path=str(pending_file)
        )

    assert success is False
    saved = json.loads(pending_file.read_text())
    assert len(saved) == 1
    assert saved[0]["content"] == "Test post"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd linkedin-bot
pytest tests/test_scheduler.py -v
```

Expected: ImportError (scheduler.py doesn't exist yet).

- [ ] **Step 3: Create `linkedin-bot/scheduler.py`**

```python
import json
import os
from datetime import datetime, timedelta, timezone

import requests
from zoneinfo import ZoneInfo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PENDING_PATH = os.path.join(BASE_DIR, "pending_posts.json")
GETLATE_BASE = "https://getlate.dev/api/v1"
IST = ZoneInfo("Asia/Kolkata")

# (weekday, hour) pairs in priority order. weekday: Mon=0 ... Sun=6
PRIORITY_SLOTS = [
    (1, 8),   # Tuesday 8 AM
    (2, 8),   # Wednesday 8 AM
    (3, 8),   # Thursday 8 AM
    (1, 17),  # Tuesday 5 PM
    (2, 17),  # Wednesday 5 PM
    (3, 17),  # Thursday 5 PM
    (0, 8),   # Monday 8 AM
    (4, 8),   # Friday 8 AM
]


def _load_pending(pending_path: str) -> list:
    if os.path.exists(pending_path):
        return json.loads(open(pending_path).read())
    return []


def _is_within_24h(slot: datetime, scheduled_posts: list) -> bool:
    """Return True if any existing post is within 24h of this slot."""
    for post in scheduled_posts:
        raw = post.get("scheduled_for", "")
        if not raw:
            continue
        try:
            existing = datetime.fromisoformat(raw)
            if existing.tzinfo is None:
                existing = existing.replace(tzinfo=IST)
            if abs((slot - existing).total_seconds()) < 86400:
                return True
        except ValueError:
            continue
    return False


def next_optimal_slot(now: datetime = None, pending_path: str = DEFAULT_PENDING_PATH) -> datetime:
    """Return the next available optimal IST posting slot."""
    if now is None:
        now = datetime.now(tz=IST)

    pending = _load_pending(pending_path)

    # Search up to 14 days forward
    for days_ahead in range(14):
        candidate_date = (now + timedelta(days=days_ahead)).date()

        for (weekday, hour) in PRIORITY_SLOTS:
            if candidate_date.weekday() != weekday:
                continue

            slot = datetime(
                candidate_date.year, candidate_date.month, candidate_date.day,
                hour, 0, 0, tzinfo=IST
            )

            # Must be at least 1 hour in the future
            if slot <= now + timedelta(hours=1):
                continue

            # Must not conflict with existing scheduled posts
            if _is_within_24h(slot, pending):
                continue

            return slot

    # Absolute fallback: next Monday 8 AM
    days_to_monday = (7 - now.weekday()) % 7 or 7
    fallback = now + timedelta(days=days_to_monday)
    return datetime(fallback.year, fallback.month, fallback.day, 8, 0, 0, tzinfo=IST)


def schedule_post(
    content: str,
    account_id: str,
    late_api_key: str,
    slot: datetime,
    pending_path: str = DEFAULT_PENDING_PATH,
) -> bool:
    """POST to GetLate. Return True on success, False on failure."""
    scheduled_for = slot.strftime("%Y-%m-%dT%H:%M:%S")

    payload = {
        "content": content,
        "scheduledFor": scheduled_for,
        "timezone": "Asia/Kolkata",
        "platforms": [{"platform": "linkedin", "accountId": account_id}],
    }

    try:
        resp = requests.post(
            f"{GETLATE_BASE}/posts",
            headers={
                "Authorization": f"Bearer {late_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        return True

    except Exception as e:
        _save_pending(content, slot, pending_path, error=str(e))
        return False


def _save_pending(content: str, slot: datetime, pending_path: str, error: str = ""):
    """Append a failed post to pending_posts.json."""
    pending = _load_pending(pending_path)
    pending.append({
        "content": content,
        "scheduled_for": slot.isoformat(),
        "error": error,
    })
    with open(pending_path, "w") as f:
        json.dump(pending, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd linkedin-bot
pytest tests/test_scheduler.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add linkedin-bot/scheduler.py linkedin-bot/tests/test_scheduler.py
git commit -m "feat: add slot picker and GetLate scheduler with pending fallback"
```

---

### Task 6: `main.py` — CLI Entry Point

**Files:**
- Create: `linkedin-bot/main.py`

No unit tests — this is the thin CLI wiring layer. Manually verify with a dry run.

- [ ] **Step 1: Create `linkedin-bot/main.py`**

```python
import sys
from config import get_env, load_context, get_account_id
from generator import generate
from scheduler import next_optimal_slot, schedule_post


def main():
    # 1. Load env vars
    env = get_env()

    # 2. Load context files + print warnings
    ctx = load_context()
    for warning in ctx["warnings"]:
        print(f"⚠️  {warning}")
    if ctx["warnings"]:
        print()

    # 3. Get topic from user
    print("📝 What's your topic or idea?")
    topic = input("> ").strip()
    if not topic:
        print("❌ No topic entered. Exiting.")
        sys.exit(1)
    print()

    # 4. Generate post (3-pass Groq)
    post, char_count = generate(
        topic=topic,
        groq_key=env["groq_key"],
        posts_text=ctx["posts_text"],
        icp_text=ctx["icp_text"],
    )

    # 5. Find next optimal slot
    slot = next_optimal_slot()
    slot_display = slot.strftime("%A %d %b, %I:%M %p IST")

    # 6. Display post for review
    print()
    print("─" * 50)
    print(f"THE POST ({char_count:,} chars)")
    print("─" * 50)
    print(post)
    print("─" * 50)
    print(f"\n📅 Will schedule for: {slot_display}\n")

    # 7. Approval
    answer = input("Approve and schedule? [y/n]: ").strip().lower()
    if answer != "y":
        print("\nPost discarded. Run again when ready.")
        sys.exit(0)

    # 8. Get LinkedIn account ID (cached after first run)
    account_id = get_account_id(late_api_key=env["late_key"])

    # 9. Schedule
    print("\n⏳ Scheduling...")
    success = schedule_post(
        content=post,
        account_id=account_id,
        late_api_key=env["late_key"],
        slot=slot,
    )

    if success:
        print(f"✅ Scheduled! Post goes live: {slot.strftime('%A %d %b %Y, %I:%M %p IST')}")
    else:
        print("❌ Scheduling failed. Post saved to pending_posts.json — reschedule manually.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a dry-run smoke test**

```bash
cd linkedin-bot
python -c "from main import main; print('main.py imports OK')"
```

Expected: `main.py imports OK` with no errors.

- [ ] **Step 3: Commit**

```bash
git add linkedin-bot/main.py
git commit -m "feat: add CLI entry point with approval flow"
```

---

### Task 7: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
cd linkedin-bot
pytest tests/ -v
```

Expected: All tests PASS. Note the count — should be 20 tests total.

- [ ] **Step 2: Fix any failures before proceeding**

If any test fails, fix the implementation (not the test) and re-run until all pass.

---

### Task 8: README

**Files:**
- Create: `linkedin-bot/README.md`

- [ ] **Step 1: Create `linkedin-bot/README.md`**

```markdown
# LinkedIn Post Generator

CLI tool to generate and schedule LinkedIn posts using Groq AI + GetLate.

## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Copy `.env.example` to `.env` and fill in your API keys:
   GROQ_API_KEY=your_groq_key
   LATE_API_KEY=your_getlate_key

3. Connect your LinkedIn account at https://getlate.dev/dashboard/connections

## Usage

   python main.py

Then enter your topic when prompted. The tool will:
- Generate 3–5 hooks and auto-select the best one
- Write a full LinkedIn post (1,300–1,600 chars)
- Show you the post for approval
- Schedule it to your LinkedIn at the next optimal IST time slot

## Optional: Improve quality over time

- Add your best LinkedIn posts to `my_posts.txt` (one post per section)
- Add your target audience description to `icp.txt`

The tool works without these files but improves significantly with them.

## Post timing (IST)

Slots are chosen in this order:
1. Tue / Wed / Thu — 8:00 AM
2. Tue / Wed / Thu — 5:00 PM
3. Mon / Fri — 8:00 AM

## Fallback

If scheduling fails, the post is saved to `pending_posts.json` for manual retry.
```

- [ ] **Step 2: Commit**

```bash
git add linkedin-bot/README.md
git commit -m "docs: add README with setup and usage instructions"
```

---

### Task 9: Final Integration Check

- [ ] **Step 1: Verify `.env` is gitignored**

```bash
cd linkedin-bot
git check-ignore .env
```

Expected: `.env` printed (meaning it is ignored).

- [ ] **Step 2: Run full test suite one last time**

```bash
cd linkedin-bot
pytest tests/ -v --tb=short
```

Expected: All tests PASS.

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: final integration check — all tests passing"
```
