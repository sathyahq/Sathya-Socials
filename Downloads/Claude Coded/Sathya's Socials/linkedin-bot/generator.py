import re

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

    if not hooks:
        raise ValueError("Pass 1 returned no hooks. Check Groq response or retry.")

    print("⚙️  Scoring hooks...", end=" ", flush=True)
    best_idx = score_hooks(client, hooks, system_prompt)
    winning_hook = hooks[best_idx]
    print(f'selected: "Hook {best_idx + 1}"')

    print("⚙️  Writing full post...")
    post = generate_post(client, topic, winning_hook, system_prompt)

    return post, len(post)
