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
