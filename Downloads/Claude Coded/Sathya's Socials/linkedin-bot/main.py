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
