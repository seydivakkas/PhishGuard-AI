"""
Agent 1 — Email Ingestion Agent
Ham e-posta verilerini sisteme alır, header/body ayrıştırma yapar.
"""
import re
from bs4 import BeautifulSoup


async def ingest_email(subject: str, body: str, sender: str = None, email_id: str = None) -> dict:
    """Ham e-posta verisini parse eder ve downstream agentlara hazırlar."""

    # HTML varsa plain text'e dönüştür
    body_plain = body
    body_html = None

    if bool(re.search(r'<[^>]+>', body)):
        body_html = body
        try:
            soup = BeautifulSoup(body, 'html.parser')
            body_plain = soup.get_text(separator=' ', strip=True)
        except Exception:
            body_plain = re.sub(r'<[^>]+>', ' ', body)

    # Basit attachment tespiti
    has_attachment = bool(re.search(
        r'(Content-Disposition:\s*attachment|filename=)', body, re.IGNORECASE
    ))

    return {
        "email_id": email_id,
        "headers": {
            "from": sender or "unknown",
            "subject": subject,
        },
        "body_plain": body_plain,
        "body_html": body_html,
        "has_attachment": has_attachment,
    }
