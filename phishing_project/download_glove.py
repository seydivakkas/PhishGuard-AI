"""
GloVe 6B 100d Embedding Indirici
Stanford NLP'den glove.6B.zip indirir ve 100d vektorleri cikarir.
"""
import os
import sys
import urllib.request
import zipfile

# Encoding fix for Windows
sys.stdout.reconfigure(encoding='utf-8')

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(TARGET_DIR, "glove.6B.zip")
TARGET_FILE = os.path.join(TARGET_DIR, "glove.6B.100d.txt")


def download_with_progress(url, dest):
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Indiriliyor: {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)")
        sys.stdout.flush()

    print(f"URL: {url}")
    urllib.request.urlretrieve(url, dest, reporthook)
    print()


def main():
    if os.path.exists(TARGET_FILE):
        size_mb = os.path.getsize(TARGET_FILE) / (1024 * 1024)
        print(f"[OK] GloVe zaten mevcut: {TARGET_FILE} ({size_mb:.1f} MB)")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print("[DOWNLOAD] GloVe 6B indiriliyor (822 MB)...")
        download_with_progress(GLOVE_URL, ZIP_PATH)
        print("[OK] Indirme tamamlandi.")
    else:
        print(f"[ZIP] ZIP zaten mevcut: {ZIP_PATH}")

    print("[EXTRACT] glove.6B.100d.txt cikariliyor...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        target_name = "glove.6B.100d.txt"
        if target_name in zf.namelist():
            zf.extract(target_name, TARGET_DIR)
            print(f"[OK] Cikarildi: {TARGET_FILE}")
        else:
            print(f"[ERROR] ZIP icinde {target_name} bulunamadi!")
            return

    os.remove(ZIP_PATH)
    print("[CLEANUP] ZIP silindi.")

    size_mb = os.path.getsize(TARGET_FILE) / (1024 * 1024)
    print(f"\n[DONE] GloVe hazir: {TARGET_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
