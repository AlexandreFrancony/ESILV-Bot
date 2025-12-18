"""
ESILV Smart Assistant - Semantic Chunker (FIXED)
================================================

Chunker for ESILV text data.
- Fixed parsing for scraper output
- Hybrid Semantic Chunking (Paragraphs > Sentences)
- Token-aware splitting (Tiktoken)
- Metadata enrichment (source, type, category)
- Saves to data/chunks.json

Date: 2025-12-16
"""

import os
import json
import logging
from typing import List, Dict, Any
import re
from datetime import datetime

import tiktoken
from tqdm import tqdm


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/chunker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger("ESILVChunker")


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "data/esilv_documents.txt"
OUTPUT_FILE = "data/chunks.json"

# Chunking parameters (tuned for Mistral 7B)
CHUNK_SIZE = 250        # Target tokens per chunk
MIN_CHUNK_SIZE = 50     # Ignore smaller chunks

# Tiktoken encoding
ENCODING_NAME = "cl100k_base"  # Used by OpenAI/Mistral models


# ============================================================================
# CHUNKER CLASS
# ============================================================================

class ESILVChunker:
    """Semantic chunker that respects document structure."""
    
    def __init__(self, input_file: str = INPUT_FILE, output_file: str = OUTPUT_FILE):
        self.input_file = input_file
        self.output_file = output_file
        self.encoder = tiktoken.get_encoding(ENCODING_NAME)
        self.chunks: List[Dict[str, Any]] = []
        self.logger = logger

    def process(self) -> bool:
        """Main processing pipeline"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Semantic Chunking")
        self.logger.info(f"Input: {self.input_file}")
        self.logger.info(f"Output: {self.output_file}")
        self.logger.info("=" * 60)
        
        # 1. Read raw file
        if not os.path.exists(self.input_file):
            self.logger.error(f"Input file not found: {self.input_file}")
            return False
            
        raw_docs = self._parse_raw_file()
        self.logger.info(f"Loaded {len(raw_docs)} source documents")
        
        if not raw_docs:
            self.logger.warning("No documents found in input file")
            return False
        
        # 2. Chunk each document
        for doc in tqdm(raw_docs, desc="Chunking documents"):
            doc_chunks = self._chunk_document(doc)
            self.chunks.extend(doc_chunks)
            
        # 3. Save to JSON
        self._save_chunks()
        
        self._log_stats()
        return True

    def _parse_raw_file(self) -> List[Dict[str, str]]:
        """
        Parse the raw text file scraped by scraper.py.
        Format:
        ================================================================================
        SOURCE: https://...
        SCRAPED: 2025-12-16...
        ================================================================================
        
        [Content here]
        """
        with open(self.input_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        documents = []
        
        # Split by SOURCE: markers (more reliable than counting equals)
        source_pattern = r"SOURCE:\s*(.+)"
        sources = list(re.finditer(source_pattern, content))
        
        for i, match in enumerate(sources):
            url = match.group(1).strip()
            
            # Get text between this SOURCE and the next one
            start_pos = match.end()
            
            if i + 1 < len(sources):
                end_pos = sources[i + 1].start()
            else:
                end_pos = len(content)
            
            section_text = content[start_pos:end_pos]
            
            # Clean: remove SCRAPED lines and separator lines
            lines = section_text.split("\n")
            clean_lines = [
                line.strip()
                for line in lines
                if (line.strip() and 
                    not line.strip().startswith("SCRAPED:") and
                    not re.match(r"^=+$", line.strip()))
            ]
            
            text = "\n".join(clean_lines).strip()
            
            if text and len(text) > 100:  # Only keep meaningful documents
                documents.append({
                    "url": url,
                    "text": text,
                    "type": self._infer_doc_type(url)
                })
                self.logger.debug(f"Parsed: {url} ({len(text)} chars)")
        
        return documents

    def _infer_doc_type(self, url: str) -> str:
        """Infer document type from URL"""
        if "formations" in url or "programs" in url:
            return "program_info"
        if "admissions" in url:
            return "admission_info"
        if "courses" in url:
            return "course_details"
        return "general_info"

    def _chunk_document(self, doc: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Split a single document into semantic chunks.
        
        Strategy:
        1. Split by double newlines (paragraphs)
        2. Combine paragraphs until chunk_size reached
        3. If single paragraph > chunk_size, split by sentences
        """
        text = doc["text"]
        url = doc["url"]
        doc_type = doc["type"]
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        chunks = []
        current_tokens = 0
        current_text = []
        
        for para in paragraphs:
            para_tokens = len(self.encoder.encode(para))
            
            # Check if adding paragraph exceeds limit
            if current_tokens + para_tokens > CHUNK_SIZE and current_text:
                # Save current chunk
                self._save_chunk(chunks, current_text, url, doc_type)
                current_tokens = 0
                current_text = []
            
            # If paragraph alone is too large, split by sentences
            if para_tokens > CHUNK_SIZE:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_tokens = len(self.encoder.encode(sent))
                    if current_tokens + sent_tokens > CHUNK_SIZE and current_text:
                        self._save_chunk(chunks, current_text, url, doc_type)
                        current_tokens = 0
                        current_text = []
                    
                    current_text.append(sent)
                    current_tokens += sent_tokens
            else:
                # Add paragraph to current chunk
                current_text.append(para)
                current_tokens += para_tokens
        
        # Save remaining content
        if current_text:
            self._save_chunk(chunks, current_text, url, doc_type)
        
        return chunks

    def _save_chunk(self, chunks_list, text_list, url, doc_type):
        """Helper to format and add chunk"""
        full_text = "\n\n".join(text_list).strip()
        num_tokens = len(self.encoder.encode(full_text))
        
        if num_tokens >= MIN_CHUNK_SIZE:
            chunks_list.append({
                "id": f"chunk_{len(self.chunks) + len(chunks_list)}",
                "text": full_text,
                "metadata": {
                    "source": url,
                    "type": doc_type,
                    "tokens": num_tokens,
                    "created_at": datetime.now().isoformat()
                }
            })

    def _save_chunks(self):
        """Save chunks to JSON file"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved {len(self.chunks)} chunks to {self.output_file}")

    def _log_stats(self):
        """Log chunking statistics"""
        if not self.chunks:
            self.logger.warning("No chunks created!")
            return
            
        total_tokens = sum(c["metadata"]["tokens"] for c in self.chunks)
        avg_tokens = total_tokens / len(self.chunks)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CHUNKING STATS")
        self.logger.info(f"   Total Chunks: {len(self.chunks)}")
        self.logger.info(f"   Total Tokens: {total_tokens}")
        self.logger.info(f"   Avg Tokens/Chunk: {avg_tokens:.1f}")
        self.logger.info(f"   Min Tokens: {min(c['metadata']['tokens'] for c in self.chunks)}")
        self.logger.info(f"   Max Tokens: {max(c['metadata']['tokens'] for c in self.chunks)}")
        self.logger.info("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    chunker = ESILVChunker()
    success = chunker.process()
    if success:
        logger.info("Chunking completed successfully!")
        return 0
    else:
        logger.error("Chunking failed.")
        return 1

if __name__ == "__main__":
    exit(main())