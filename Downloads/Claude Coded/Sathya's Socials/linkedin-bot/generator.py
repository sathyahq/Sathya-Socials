import re

from openai import OpenAI

from linkedin_rules import LINKEDIN_RULES
from context_fetcher import fetch_context

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"

FORMAT_RULES = """
## PARAGRAPH GROUPING (follow exactly)
- Blank lines go BETWEEN blocks, not between every single sentence
- A list intro line and its items stay together — no blank line between them
- Two tightly related sentences (setup + punch, cause + effect, pronoun reference) can share a block with no blank line between them
- All other standalone sentences get blank lines on both sides
- Use → for lists of 3+ parallel items (features, steps, results)
- Use x to negate items and - for the positive contrast (e.g. "x Not views.\\nx Not subscribers.\\n- Revenue.")
- Use - for action-oriented lists ("- You need to know...", "- You need to show up...")
- NEVER use em-dashes (—). Replace with a comma or rewrite.
"""

def _pre_pass(text: str) -> str:
    """Python pre-pass: strip em-dashes, collapse double spaces."""
    text = text.replace("—", ",")
    text = re.sub(r" {2,}", " ", text)
    return text


def _post_pass(formatted: str, original: str) -> str:
    """Python post-pass: assert no em-dashes, check for excessive length drift."""
    if "—" in formatted:
        raise ValueError("em-dash found in formatted post — pre-pass or LLM re-introduced it")
    drift = abs(len(formatted) - len(original)) / max(len(original), 1)
    if drift > 0.15:
        raise ValueError(
            f"length drift {drift:.0%} exceeds 15% threshold — LLM may have rewritten content"
        )
    return formatted


def _llm_format(client, post: str) -> str:
    """Pass 4: LLM-based formatting — grouping and → list conversion only."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": FORMAT_RULES},
            {
                "role": "user",
                "content": (
                    "Reformat this LinkedIn post. Do NOT change any words, sentences, or meaning.\n"
                    "Fix ONLY the formatting:\n\n"
                    "1. Apply the paragraph grouping rules exactly.\n"
                    "2. Convert any 3+ consecutive sentences with the same grammatical pattern "
                    "(e.g. 'I'm X. I'm Y. I'm Z.') into a → list.\n"
                    "3. Keep x / - contrast blocks together with no blank lines between items.\n\n"
                    "Return ONLY the reformatted post. No explanation.\n\n"
                    f"POST:\n{post}"
                ),
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def format_post(client, post: str) -> str:
    """Orchestrate Pass 4: pre-pass → LLM format → post-pass."""
    cleaned = _pre_pass(post)
    formatted = _llm_format(client, cleaned)
    return _post_pass(formatted, post)


DAY_TONE = {
    "Monday": (
        "Tone: Confident and results-focused. Lead with real client outcomes or business proof. "
        "This is SellonTube positioning day — make the reader feel the cost of not using YouTube for leads."
    ),
    "Tuesday": (
        "Tone: Curious and experimental. Write as a founder figuring things out in public. "
        "Frame AI tools through the lens of what they enabled — leads, speed, decisions — not the tool itself."
    ),
    "Wednesday": (
        "Tone: Thoughtful and contrarian. Challenge a common assumption about business or building. "
        "Back it with personal reasoning. Invite debate without baiting for engagement."
    ),
    "Thursday": (
        "Tone: Tactical and specific. Share a framework, a mistake, or a counter-intuitive YouTube insight. "
        "Make it immediately actionable. This is the most shareable and saveable post of the week."
    ),
    "Friday": (
        "Tone: Personal, reflective, and vulnerable. Start with a real life story or observation. "
        "Draw out a wellbeing insight — happiness, gratitude, or energy. Connect it to a business lesson. "
        "End with something that makes the reader feel seen, not just informed."
    ),
}


def build_system_prompt(posts_text: str, icp_text: str, live_context: str = "") -> str:
    """Build the system prompt injected into every Groq call."""
    prompt = f"## LINKEDIN BEST PRACTICES\n{LINKEDIN_RULES}\n\n"

    if posts_text:
        prompt += f"## VOICE SAMPLES (write exactly like these)\n{posts_text}\n\n"

    if icp_text:
        prompt += f"## TARGET AUDIENCE\n{icp_text}\n\n"

    if live_context:
        prompt += f"{live_context}\n\n"

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
    return 0


def generate_post(client, topic: str, winning_hook: str, system_prompt: str, day: str = "") -> str:
    """Pass 3: Generate full LinkedIn post using the winning hook as the opener."""
    tone_instruction = DAY_TONE.get(day, "")
    tone_block = f"DAY TONE ({day}):\n{tone_instruction}\n\n" if tone_instruction else ""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{tone_block}"
                    f"Topic: {topic}\n\n"
                    f"Use EXACTLY this as your first line (do not change it):\n{winning_hook}\n\n"
                    "Now write the full LinkedIn post.\n"
                    "Requirements:\n"
                    "- 1,300–1,600 characters total\n"
                    "- Max 12 words per sentence\n"
                    "- Include a second 'rehook' on line 2 to build tension\n"
                    "- End with ONE clear CTA (no engagement bait)\n"
                    "- 0–3 hashtags max, placed at the end\n"
                    "- No external links in the post body\n"
                    "- No generic AI phrases\n\n"
                    "FORMATTING RULES (critical — follow exactly):\n"
                    "- NEVER use em-dashes (—). Use a comma or rewrite instead\n"
                    "- Blank lines go BETWEEN blocks, not between every single sentence\n"
                    "- A list intro line and its items stay together with NO blank line between them\n"
                    "- Two tightly related sentences (setup + punch) can share a block with no blank line\n"
                    "- Standalone sentences get blank lines on both sides\n"
                    "- Use → for lists of 3+ parallel items (features, results, steps)\n"
                    "- Use x to negate and - for the positive contrast: 'x Not views.\\nx Not subscribers.\\n- Revenue.'\n"
                    "- Use - for action lists: '- You need to know X.\\n- You need to show up.'\n\n"
                    "AVOID THESE PHRASES (rewrite in Sathya's natural voice):\n"
                    "- 'making adjustments to optimize' → 'testing what works'\n"
                    "- 'pipeline for potential customers' → 'way to bring in buyers'\n"
                    "- 'designed to convert' → 'built to get leads'\n"
                    "- 'I\\'m analyzing what works and what doesn\\'t' → 'I track what converts. I cut what doesn\\'t.'\n"
                    "- 'content for entertainment' → 'content that does nothing'\n"
                    "- 'specific call to action' → 'one clear next step'\n"
                    "- 'optimize the results' → 'improve what I see'\n"
                    "- 'tracking the conversion rate' → 'watching what converts'\n\n"
                    "Output ONLY the post. No explanations, no preamble."
                ),
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def generate(topic: str, day: str, groq_key: str, posts_text: str, icp_text: str) -> tuple[str, int]:
    """Orchestrate all 3 passes. Return (post_text, char_count)."""
    client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)

    # Fetch live context for Mon/Thu before building prompt
    live_context = fetch_context(day)
    system_prompt = build_system_prompt(posts_text, icp_text, live_context)

    print("⚙️  Generating hooks...")
    hooks = generate_hooks(client, topic, system_prompt)

    if not hooks:
        raise ValueError("Pass 1 returned no hooks. Check Groq response or retry.")

    print("⚙️  Scoring hooks...", end=" ", flush=True)
    best_idx = score_hooks(client, hooks, system_prompt)
    winning_hook = hooks[best_idx]
    print(f'selected: "Hook {best_idx + 1}"')

    print("⚙️  Writing full post...")
    post = generate_post(client, topic, winning_hook, system_prompt, day=day)

    print("⚙️  Formatting post...")
    post = format_post(client, post)

    return post, len(post)
