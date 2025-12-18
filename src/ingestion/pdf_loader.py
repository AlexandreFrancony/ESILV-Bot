"""
ESILV Smart Assistant - PDF Loader
==================================

Production-ready PDF loader for ESILV brochure.
- Extracts text from PDF files
- Handles large files efficiently
- Preserves document structure
- Saves to data/esilv_documents.txt (compatible with chunker.py)

Date: 2025-12-18
"""

import os
import logging
from typing import List, Dict, Any
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
        fh = logging.FileHandler(f"logs/pdf_loader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger("ESILVPDFLoader")


# ============================================================================
# CONFIGURATION
# ============================================================================

PDF_PATH = "data/esilv_brochure.pdf"
OUTPUT_FILE = "data/esilv_documents.txt"
CHUNK_SIZE = 500  # Characters per chunk (before chunker.py processes)


# ============================================================================
# PDF LOADER CLASS
# ============================================================================

class PDFLoader:
    """
    PDF loader that extracts text from PDF files.
    
    Supports two extraction methods:
    1. pdfplumber (better text preservation, recommended)
    2. PyPDF2 (fallback if pdfplumber not available)
    """
    
    def __init__(self, pdf_path: str = PDF_PATH, output_file: str = OUTPUT_FILE):
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
        """Main loading pipeline"""
        self.logger.info("=" * 60)
        self.logger.info("Starting PDF Loading")
        self.logger.info(f"PDF file: {self.pdf_path}")
        self.logger.info(f"Output file: {self.output_file}")
        self.logger.info("=" * 60)
        
        # 1. Verify file exists
        if not os.path.exists(self.pdf_path):
            self.logger.error(f"PDF file not found: {self.pdf_path}")
            return False
        
        # Check file size
        file_size_mb = os.path.getsize(self.pdf_path) / (1024 * 1024)
        self.logger.info(f"PDF file size: {file_size_mb:.2f} MB")
        
        # 2. Extract text
        if self.extraction_method == "pdfplumber":
            success = self._extract_with_pdfplumber()
        else:
            success = self._extract_with_pypdf2()
        
        if not success:
            return False
        
        # 3. Save to output file
        self._save_output()
        
        # 4. Log statistics
        self._log_stats()
        
        return True
    
    def _extract_with_pdfplumber(self) -> bool:
        """Extract text using pdfplumber (recommended)"""
        try:
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
                            self.logger.info(f"Extracted {i + 1}/{self.total_pages} pages")
                    
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
                            self.logger.info(f"Extracted {i + 1}/{self.total_pages} pages")
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to extract page {i + 1}: {e}")
                        continue
            
            self.logger.info(f"Successfully extracted {self.total_pages} pages")
            return True
        
        except Exception as e:
            self.logger.error(f"PyPDF2 extraction error: {e}")
            return False
    
    def _save_output(self):
        """Save extracted text to output file"""
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
            
            # Format as single document (compatible with chunker.py)
            output_content = f"""ESILV Smart Assistant - Extracted Documents
Generated: {datetime.now().isoformat()}

================================================================================
SOURCE: ESILV Brochure PDF
SCRAPED: {datetime.now().isoformat()}
================================================================================

{self.extracted_text}

================================================================================
END OF DOCUMENT
================================================================================
"""
            
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(output_content)
            
            self.logger.info(f"Saved extracted text to {self.output_file}")
        
        except Exception as e:
            self.logger.error(f"Error saving output: {e}")
            raise
    
    def _log_stats(self):
        """Log extraction statistics"""
        char_count = len(self.extracted_text)
        word_count = len(self.extracted_text.split())
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXTRACTION STATS")
        self.logger.info(f"   Total Pages: {self.total_pages}")
        self.logger.info(f"   Total Characters: {char_count:,}")
        self.logger.info(f"   Total Words: {word_count:,}")
        self.logger.info(f"   Avg Chars/Page: {char_count / self.total_pages if self.total_pages else 0:.0f}")
        self.logger.info("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    logger.info("Starting PDF Loader")
    
    loader = PDFLoader()
    success = loader.load()
    
    if success:
        logger.info("PDF loading completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Run chunker: python src/ingestion/chunker.py")
        logger.info("2. Reindex: python src/ingestion/indexer.py")
        return 0
    else:
        logger.error("PDF loading failed.")
        return 1


if __name__ == "__main__":
    exit(main())
