"""
Agent 2 — Parallel Preprocessing Agent
Metin temizleme ve SHA-256 hash hesaplamasını paralel gerçekleştirir.
"""
import asyncio
import hashlib
import re


async def clean_text(text: str) -> str:
    """Metin temizleme — Thread A."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)           # URL kaldır
    text = re.sub(r'\S+@\S+', '', text)            # E-posta kaldır
    text = re.sub(r'<[^>]+>', '', text)             # HTML tag kaldır
    text = re.sub(r'[^a-z\s]', '', text)            # Özel karakter kaldır
    text = re.sub(r'\s+', ' ', text).strip()        # Boşluk normalize
    return text


async def compute_sha256(text: str) -> str:
    """SHA-256 hash hesaplama — Thread B."""
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()


# Duplicate cache (runtime boyunca)
_seen_hashes: set = set()


async def parallel_preprocess(body: str, subject: str) -> dict:
    """
    Paralel ön işleme — asyncio.gather ile eşzamanlı çalıştırır.
    Thread A: Metin temizleme
    Thread B: SHA-256 hash (body + subject)
    """
    cleaned, sha256_body, sha256_subject = await asyncio.gather(
        clean_text(body),
        compute_sha256(body),
        compute_sha256(subject)
    )

    # Duplicate tespiti
    is_duplicate = sha256_body in _seen_hashes
    if not is_duplicate:
        _seen_hashes.add(sha256_body)

    return {
        "cleaned_text": cleaned,
        "sha256_body": sha256_body,
        "sha256_subject": sha256_subject,
        "is_duplicate": is_duplicate,
    }
