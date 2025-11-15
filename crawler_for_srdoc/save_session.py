# save_session.py
from playwright.sync_api import sync_playwright

START_URL = "https://mysu.sabanciuniv.edu/surecharitasi/tr/surec-tablosu"
STATE_FILE = r"C:\\Users\\kosot\\OneDrive\\Masaüstü\\CS\\bitirme\\crawler_for_srdoc\\session.json"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # visible so you can MFA
    context = browser.new_context()
    page = context.new_page()
    page.goto(START_URL, wait_until="load")
    print(">>> Please complete CAS login in the browser window.")
    page.wait_for_timeout(15_000)  # give yourself time; extend if needed
    context.storage_state(path=STATE_FILE)
    print(f"Saved storage state to {STATE_FILE}")
    browser.close()
