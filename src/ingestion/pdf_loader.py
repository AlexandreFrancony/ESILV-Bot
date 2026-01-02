"""
ESILV Smart Assistant - PDF Loader
==================================

Multi-PDF loader for ESILV documents.
- Extracts text from several PDF files
- Handles large files efficiently
- Preserves document structure
- Saves to data/esilv_documents.txt (compatible with chunker.py)

Date: 2025-12-18 (updated 2026-01-02)
"""

import os
import logging
from datetime import datetime

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(
            f"logs/pdf_loader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


logger = setup_logger("ESILVPDFLoader")


# ============================================================================
# CONFIGURATION
# ============================================================================

PDF_FILES = [
    ("ESILV Brochure PDF", "data/esilv_brochure.pdf"),
    ("ESILV MSc PDF", "data/esilv-msc.pdf"),
    ("ESILV Bachelors PDF", "data/bachelors-esilv.pdf"),
    ("ESILV Apprentissage PDF", "data/plaquette_apprentissage_esilv.pdf"),
    ("ESILV Livret pédagogique PDF", "data/livret_pedagogique_esilv.pdf"),
]

OUTPUT_FILE = "data/esilv_documents.txt"
CHUNK_SIZE = 500  # Characters per chunk (before chunker.py processes)


# ============================================================================
# PDF LOADER CLASS (single PDF)
# ============================================================================

class PDFLoader:
    """
    PDF loader that extracts text from ONE PDF file.

    Supports two extraction methods:
    1. pdfplumber (better text preservation, recommended)
    2. PyPDF2 (fallback if pdfplumber not available)
    """

    def __init__(self, pdf_path: str, output_file: str = OUTPUT_FILE):
        self.pdf_path = pdf_path
        self.output_file = output_file
        self.logger = logger
        self.total_pages = 0
        self.extracted_text = ""

        # Check dependencies
        self._check_dependencies()

    def _check_dependencies(self):
        """Verify PDF extraction libraries are available"""
        if not pdfplumber and not PyPDF2:
            self.logger.error("Neither pdfplumber nor PyPDF2 installed")
            self.logger.error("Install with: pip install pdfplumber PyPDF2")
            raise ImportError("PDF extraction library required")

        if pdfplumber:
            self.logger.info("Using pdfplumber for PDF extraction")
            self.extraction_method = "pdfplumber"
        else:
            self.logger.info("Using PyPDF2 for PDF extraction")
            self.extraction_method = "PyPDF2"

    def load(self) -> bool:
        """Main loading pipeline for one PDF"""
        self.logger.info("=" * 60)
        self.logger.info("Starting PDF Loading")
        self.logger.info(f"PDF file: {self.pdf_path}")
        self.logger.info("=" * 60)

        if not os.path.exists(self.pdf_path):
            self.logger.error(f"PDF file not found: {self.pdf_path}")
            return False

        file_size_mb = os.path.getsize(self.pdf_path) / (1024 * 1024)
        self.logger.info(f"PDF file size: {file_size_mb:.2f} MB")

        if self.extraction_method == "pdfplumber":
            success = self._extract_with_pdfplumber()
        else:
            success = self._extract_with_pypdf2()

        if not success:
            return False

        self._log_stats()
        return True

    def _extract_with_pdfplumber(self) -> bool:
        """Extract text using pdfplumber (recommended)"""
        try:
            if not pdfplumber:
                self.logger.error("pdfplumber is not available")
                return False

            self.logger.info("Extracting text with pdfplumber...")

            with pdfplumber.open(self.pdf_path) as pdf:
                self.total_pages = len(pdf.pages)
                self.logger.info(f"Total pages: {self.total_pages}")

                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            self.extracted_text += f"\n--- Page {i + 1} ---\n{text}\n"

                        if (i + 1) % 10 == 0:
                            self.logger.info(
                                f"Extracted {i + 1}/{self.total_pages} pages"
                            )

                    except Exception as e:
                        self.logger.warning(f"Failed to extract page {i + 1}: {e}")
                        continue

            self.logger.info(f"Successfully extracted {self.total_pages} pages")
            return True

        except Exception as e:
            self.logger.error(f"pdfplumber extraction error: {e}")
            return False

    def _extract_with_pypdf2(self) -> bool:
        """Extract text using PyPDF2 (fallback)"""
        try:
            if not PyPDF2:
                self.logger.error("PyPDF2 is not available")
                return False

            self.logger.info("Extracting text with PyPDF2...")

            with open(self.pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                self.total_pages = len(pdf_reader.pages)
                self.logger.info(f"Total pages: {self.total_pages}")

                for i, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            self.extracted_text += f"\n--- Page {i + 1} ---\n{text}\n"

                        if (i + 1) % 10 == 0:
                            self.logger.info(
                                f"Extracted {i + 1}/{self.total_pages} pages"
                            )

                    except Exception as e:
                        self.logger.warning(f"Failed to extract page {i + 1}: {e}")
                        continue

            self.logger.info(f"Successfully extracted {self.total_pages} pages")
            return True

        except Exception as e:
            self.logger.error(f"PyPDF2 extraction error: {e}")
            return False

    def _log_stats(self):
        """Log extraction statistics"""
        char_count = len(self.extracted_text)
        word_count = len(self.extracted_text.split())

        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXTRACTION STATS")
        self.logger.info(f"   Total Pages: {self.total_pages}")
        self.logger.info(f"   Total Characters: {char_count:,}")
        self.logger.info(f"   Total Words: {word_count:,}")
        self.logger.info(
            f"   Avg Chars/Page: {char_count / self.total_pages if self.total_pages else 0:.0f}"
        )
        self.logger.info("=" * 60)


# ============================================================================
# MAIN EXECUTION (multi-PDF)
# ============================================================================

def main():
    logger.info("Starting PDF Loader (multi-PDF)")

    all_text_blocks = []

    for source_name, pdf_path in PDF_FILES:
        logger.info(f"--- Loading {source_name}: {pdf_path} ---")
        loader = PDFLoader(pdf_path=pdf_path, output_file=OUTPUT_FILE)
        success = loader.load()
        if not success:
            logger.error(f"Failed to load {pdf_path}, skipping.")
            continue

        block = f"""
================================================================================
SOURCE: {source_name}
SCRAPED: {datetime.now().isoformat()}
FILE: {pdf_path}
================================================================================

{loader.extracted_text}

"""
        all_text_blocks.append(block)

    if not all_text_blocks:
        logger.error("No PDF was successfully loaded.")
        return 1

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)

    output_content = (
        "ESILV Smart Assistant - Extracted Documents (Multi-PDF)\n"
        f"Generated: {datetime.now().isoformat()}\n\n"
        + "\n".join(all_text_blocks)
        + "\n================================================================================\n"
        + "END OF DOCUMENTS\n"
        + "================================================================================\n"
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_content)

    logger.info(f"Saved merged extracted text to {OUTPUT_FILE}")
    logger.info("Next steps:")
    logger.info("1. Run chunker: python src/ingestion/chunker.py")
    logger.info("2. Reindex:     python src/ingestion/indexer.py")
    return 0


if __name__ == "__main__":
    exit(main())
