import re

from openai import OpenAI

from linkedin_rules import LINKEDIN_RULES

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
        max_tokens=1200,
        temperature=0.8,
        messages=[
            {
                "role": "user",
                "content": (
                    "Here is an example of a well-written LinkedIn post. Study its length and depth:\n\n"
                    "---EXAMPLE START---\n"
                    "My 9-year-old just made Google feel ancient.\n\n"
                    "We were watching Sookshmadarshini, the Malayalam film with Nazriya.\n\n"
                    "A character mentioned Alzheimer's.\n\n"
                    "She paused. Thought for a second. Then reached for her phone.\n\n"
                    "I assumed she'd Google it.\n\n"
                    "But no.\n\n"
                    "She opened ChatGPT.\n\n"
                    "That moment stayed with me longer than the movie did.\n\n"
                    "I grew up with 'just Google it.'\n\n"
                    "For her? Google was never the reflex.\n\n"
                    "She didn't want links. She wanted answers.\n\n"
                    "No sifting through ten blue links.\n\n"
                    "No ads dressed up as content.\n\n"
                    "No SEO-padded fluff.\n\n"
                    "Just a direct conversation with something that knows things.\n\n"
                    "Here's what that unlocked for me.\n\n"
                    "The shift isn't about Google losing.\n\n"
                    "It's about people raising their expectations.\n\n"
                    "They don't want information. They want understanding.\n\n"
                    "And if you're still creating content like it's 2015, generic and safe, "
                    "AI will answer that question before your article even loads.\n\n"
                    "What wins now:\n\n"
                    "Real experience, not recycled advice.\n\n"
                    "Specific stories, not broad claims.\n\n"
                    "Depth that no AI can fake.\n\n"
                    "My daughter didn't choose ChatGPT over Google.\n\n"
                    "She chose answers over links.\n\n"
                    "Are you still giving people links?\n"
                    "---EXAMPLE END---\n\n"
                    f"Now write a NEW post about: {topic}\n\n"
                    f"Start with this exact line: {winning_hook}\n\n"
                    "Match the example's length and depth exactly.\n"
                    "Apply this exact formatting logic:\n"
                    "1. HOOK: always a solo line.\n"
                    "2. PIVOT/REVELATION (single turning-point insight): solo line, blank line before and after.\n"
                    "3. NARRATIVE GROUP (2-3 sentences that flow together as one continuous mini-story): group them with NO blank line between them, blank line after the group.\n"
                    "4. PARALLEL LIST (3+ sentences with the same grammatical pattern): convert to a `-` bullet list under a lead-in sentence. Use plain `-` bullets only.\n"
                    "5. Do NOT put every sentence on its own paragraph. Group first, then separate groups with blank lines.\n"
                    "Short sentences. First person. Specific details. No hashtags. No emojis. No em-dashes.\n"
                    "End with one direct question.\n\n"
                    "Output ONLY the post. Nothing else."
                ),
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def generate(topic: str, groq_key: str, posts_text: str, icp_text: str) -> tuple[str, int]:
    """Orchestrate all 4 passes. Return (post_text, char_count)."""
    client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)
    system_prompt = build_system_prompt(posts_text, icp_text)

    print("Generating hooks...")
    hooks = generate_hooks(client, topic, system_prompt)

    if not hooks:
        raise ValueError("Pass 1 returned no hooks. Check Groq response or retry.")

    print("Scoring hooks...", end=" ", flush=True)
    best_idx = score_hooks(client, hooks, system_prompt)
    winning_hook = hooks[best_idx]
    print(f'selected: Hook {best_idx + 1}')

    print("Writing full post...")
    post = generate_post(client, topic, winning_hook, system_prompt)

    print("Formatting post...")
    post = format_post(client, post)

    return post, len(post)
