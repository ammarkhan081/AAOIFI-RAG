"""Isolate basic PDF-open and first-page extraction for SS9 only."""

from pathlib import Path
import sys

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIR = ROOT / "data" / "private"
SS9_PATH = PRIVATE_DIR / "SS-9-Ijarah-and-Ijarah-Muntahia-Bittamleek.pdf"
SS8_PATH = PRIVATE_DIR / "SS-8-Murabahah.pdf"


def main() -> None:
    print(f"SS9 path: {SS9_PATH}")
    print(f"SS9 size bytes: {SS9_PATH.stat().st_size}")
    print(f"SS8 size bytes: {SS8_PATH.stat().st_size}")
    with pdfplumber.open(SS9_PATH, unicode_norm="NFC") as pdf:
        print(f"SS9 page count: {len(pdf.pages)}")
        page_one_text = pdf.pages[0].extract_text() or ""
        print(f"SS9 page 1 extracted text length: {len(page_one_text)}")


if __name__ == "__main__":
    main()
