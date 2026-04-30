"""Quick sanity check for Gemini judge: verifies response_mime_type works."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'c:\bitirme3\ENS491-MACKIS')
from dotenv import load_dotenv
load_dotenv(r'c:\bitirme3\ENS491-MACKIS\.env')

key = os.getenv('GEMINI_API_KEY', '')
if not key:
    print('ERROR: No GEMINI_API_KEY in .env')
    sys.exit(1)
print(f'Key found: {key[:10]}...')

from google import genai

client = genai.Client(api_key=key)
prompt = 'Return only a JSON object with these exact fields: {"score": 8, "reasoning": "test passed"}'

# Test WITH response_mime_type (forces JSON, no markdown)
print('\n--- Test WITH response_mime_type=application/json ---')
try:
    resp = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=64,
            response_mime_type='application/json',
        ),
    )
    print('RAW:', repr(resp.text[:200]))
except Exception as e:
    print(f'FAILED: {e}')

    # Fallback test WITHOUT response_mime_type
    print('\n--- Test WITHOUT response_mime_type (fallback) ---')
    try:
        resp2 = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=64,
            ),
        )
        print('RAW:', repr(resp2.text[:200]))
    except Exception as e2:
        print(f'ALSO FAILED: {e2}')
