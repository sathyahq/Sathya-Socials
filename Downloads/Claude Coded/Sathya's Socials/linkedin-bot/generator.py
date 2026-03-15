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
                    "Each sentence on its own line with a blank line between each.\n"
                    "Short sentences. First person. Specific details. No hashtags. No emojis. No em-dashes.\n"
                    "End with one direct question.\n\n"
                    "Output ONLY the post. Nothing else."
                ),
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def generate(topic: str, groq_key: str, posts_text: str, icp_text: str) -> tuple[str, int]:
    """Orchestrate all 3 passes. Return (post_text, char_count)."""
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

    return post, len(post)
