# LinkedIn Post Generator & Scheduler — Design Spec
**Date:** 2026-03-15
**Branch:** Sathyas-LinkedIn

---

## Overview

A Python CLI tool that accepts a topic, generates an optimised LinkedIn post via Groq (3-pass), auto-selects the best hook, shows the post for approval, then schedules it to LinkedIn via GetLate at the next optimal IST time slot.

---

## Architecture & Data Flow

```
CLI prompt (topic)
        │
        ▼
  generator.py
  ├─ Pass 1: Generate 3–5 hooks        ──► Groq (llama-3.3-70b-versatile)
  ├─ Pass 2: Score & pick best hook    ──► Groq (evaluator call)
  └─ Pass 3: Generate full post        ──► Groq (winning hook as fixed opener)
        │
        ▼
  main.py — display post in terminal
        │
   user approves? [y/n]
   ├─ No  → exit
   └─ Yes → scheduler.py
                ├─ Find next optimal IST slot
                ├─ POST /posts to GetLate API
                ├─ Success → confirm to user
                └─ Fail    → save to pending_posts.json
```

Every Groq call receives the same system prompt:
- Full `LINKEDIN_RULES` constant from `linkedin_rules.py`
- Contents of `my_posts.txt` (if non-empty) — voice samples
- Contents of `icp.txt` (if non-empty) — target audience

---

## Components

| File | Responsibility |
|---|---|
| `main.py` | CLI entry, topic prompt, display post, approval loop |
| `generator.py` | 3 Groq passes: hooks → score → full post |
| `scheduler.py` | Next optimal IST slot calc + GetLate POST, pending fallback |
| `config.py` | Load `.env`, read/write `config.json`, fetch+cache `accountId` on first run |
| `linkedin_rules.py` | Single `LINKEDIN_RULES` constant (full best practices text) |
| `my_posts.txt` | Voice samples — filled in manually by user |
| `icp.txt` | Target audience — filled in manually by user |
| `config.json` | Cached `accountId` (auto-created on first run) |
| `pending_posts.json` | Failed schedules saved here for retry |
| `.env` | `GROQ_API_KEY`, `LATE_API_KEY` |
| `requirements.txt` | `openai`, `requests`, `python-dotenv` |

---

## Generator Detail (3-Pass Groq)

**Pass 1 — Hook generation**
- Prompt: topic + rules + voice samples + ICP
- Output: 3–5 hook variations (numbered list)
- Each hook must be under 210 characters (before "See More" cutoff)

**Pass 2 — Hook scoring**
- Prompt: all hooks + scoring criteria from linkedin_rules.py
- Criteria: dwell time potential, hook type match, no banned phrases, character count, surprise factor
- Output: index of best hook + one-line reason

**Pass 3 — Full post generation**
- Prompt: winning hook (fixed) + topic + rules + voice samples + ICP
- Instruction: "Use exactly this hook as your first line. Do not change it."
- Output: complete LinkedIn post (1,300–1,600 chars target)

---

## Scheduler Detail

**Optimal IST slot priority order:**
1. Tue / Wed / Thu — 8:00 AM IST (primary)
2. Tue / Wed / Thu — 5:00 PM IST (secondary)
3. Mon / Fri — 8:00 AM IST (fallback)

**Slot picker algorithm:**
1. Get current IST datetime
2. Walk forward through upcoming days
3. For each candidate slot: check it is >1h in the future
4. Check no other post is already scheduled within 24h (via `pending_posts.json`)
5. Return the first passing slot

**GetLate API call:**
```
POST https://getlate.dev/api/v1/posts
Authorization: Bearer <LATE_API_KEY>
{
  "content": "<post text>",
  "scheduledFor": "2026-03-17T08:00:00",
  "timezone": "Asia/Kolkata",
  "platforms": [{"platform": "linkedin", "accountId": "<accountId>"}]
}
```

**First-run account setup:**
```
GET https://getlate.dev/api/v1/accounts
```
Find the entry where `platform == "linkedin"`, save `id` to `config.json`. Reuse on all subsequent runs.

---

## Terminal UX Flow

```
$ python main.py

📝 What's your topic or idea?
> The mistake I made trusting AI tools blindly

⚙️  Generating hooks...
⚙️  Scoring hooks... selected: "Hook 3"
⚙️  Writing full post...

──────────────────────────────────────────
THE POST (1,412 chars)
──────────────────────────────────────────
I trusted an AI tool with my client's content strategy.

It cost me a ₹2L project.

[full post body]

──────────────────────────────────────────
📅 Will schedule for: Tuesday 18 Mar, 8:00 AM IST

Approve and schedule? [y/n]: y

✅ Scheduled! Post goes live: Tuesday 18 Mar 2026, 8:00 AM IST
```

**Empty file warnings (non-blocking):**
```
⚠️  my_posts.txt is empty — generating without voice samples.
⚠️  icp.txt is empty — generating without ICP context.
```

**Scheduling failure:**
```
❌ Scheduling failed: <error message>
💾 Saved to pending_posts.json — reschedule manually.
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| `my_posts.txt` empty | Warn + continue without voice samples |
| `icp.txt` empty | Warn + continue without ICP |
| No LinkedIn account in GetLate | Error: "Connect LinkedIn at getlate.dev/dashboard/connections" |
| GetLate POST fails | Save to `pending_posts.json`, show error |
| Groq API error | Show error message, exit cleanly |
| `config.json` missing | Auto-created on first run |

---

## File Structure

```
linkedin-bot/
├── main.py
├── generator.py
├── scheduler.py
├── config.py
├── linkedin_rules.py
├── my_posts.txt
├── icp.txt
├── config.json          # gitignored
├── pending_posts.json   # gitignored
├── .env                 # gitignored
├── requirements.txt
└── README.md
```

---

## Dependencies

```
openai
requests
python-dotenv
```
