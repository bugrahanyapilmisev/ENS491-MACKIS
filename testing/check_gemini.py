"""
check_gemini.py - Quick Gemini API connectivity check

Run:
    python testing/check_gemini.py

Checks:
  1. google-genai SDK installed
  2. GEMINI_API_KEY set in .env (not a placeholder)
  3. Real API call succeeds — auto-tries candidate models in order
     and reports which one works, so you can update JUDGE_MODEL in .env
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Reading GEMINI_API_KEY from system env only.")

# ── Check 1: SDK installed ────────────────────────────────────────────────────
print("=" * 60)
print("  MACKIS — Gemini Judge Connectivity Check")
print("=" * 60)

try:
    from google import genai
    from google.genai import types as genai_types
    print("✅ [1/3] google-genai SDK is installed")
except ImportError:
    print("❌ [1/3] google-genai SDK NOT installed.")
    print("         Fix: pip install google-genai")
    sys.exit(1)

# ── Check 2: API key ──────────────────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key or api_key == "your_gemini_api_key_here":
    print("❌ [2/3] GEMINI_API_KEY is missing or still a placeholder.")
    print("         Fix: add GEMINI_API_KEY=<your_key> to your .env file")
    print("         Get a key at: https://aistudio.google.com/app/apikey")
    sys.exit(1)

masked = api_key[:8] + "..." + api_key[-4:]
print(f"✅ [2/3] GEMINI_API_KEY found: {masked}")

# ── Check 3: Real API call — try multiple models ─────────────────────────────
# Models ordered from newest free-tier-available to older fallbacks.
# update this list as Google updates the free tier.
configured_model = os.getenv("JUDGE_MODEL", "gemini-2.0-flash")

CANDIDATE_MODELS = [
    configured_model,          # whatever is currently in .env
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]
# De-duplicate while preserving order
seen = set()
CANDIDATE_MODELS = [m for m in CANDIDATE_MODELS if not (m in seen or seen.add(m))]

print(f"   Configured model : {configured_model}")
print(f"   Trying {len(CANDIDATE_MODELS)} candidate model(s)…")
print()

client = genai.Client(api_key=api_key)
working_model = None

for model in CANDIDATE_MODELS:
    sys.stdout.write(f"   Testing {model:<45s} … ")
    sys.stdout.flush()
    try:
        resp = client.models.generate_content(
            model=model,
            contents="Reply with exactly one word: READY",
            config=genai_types.GenerateContentConfig(
                max_output_tokens=8,
                temperature=0.0,
            ),
        )
        reply = resp.text.strip() if resp.text else ""
        print(f"✅  replied: \"{reply}\"")
        working_model = model
        break
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            print("⚠️  quota exceeded / free tier limit 0")
        elif "404" in err_str or "not found" in err_str.lower():
            print("⚠️  model not found")
        elif "403" in err_str or "permission" in err_str.lower():
            print("❌  permission denied (bad key?)")
            break
        else:
            print(f"❌  error: {err_str[:80]}")

print()
print("=" * 60)

if working_model:
    print(f"  🎉 Gemini judge is READY!")
    print(f"     Working model : {working_model}")
    if working_model != configured_model:
        print()
        print(f"  ⚠️  Your .env has JUDGE_MODEL={configured_model}")
        print(f"     but that model hit a quota limit.")
        print(f"  👉 Update your .env:")
        print(f"       JUDGE_MODEL={working_model}")
    print("=" * 60)
else:
    print("  ❌ No Gemini model responded successfully.")
    print()
    print("  Possible causes:")
    print("  • All models hit free-tier quota (limit: 0)")
    print("    → Enable billing on your Google Cloud project:")
    print("      https://console.cloud.google.com/billing")
    print("  • Your API key may be from a restricted project")
    print("    → Check: https://aistudio.google.com/app/apikey")
    print("  • Quotas reset at midnight Pacific Time (PT) —")
    print("    try again tomorrow if you hit the daily limit")
    print("=" * 60)
    sys.exit(1)
