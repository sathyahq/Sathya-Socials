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
