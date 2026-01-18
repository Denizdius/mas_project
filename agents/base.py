"""
Base utilities shared across all agents.

Provides:
- LLM configuration for Qwen3-8B via vLLM
- Output cleaning (remove <think> blocks)
- Configuration dataclass
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI

# --- Configuration ---
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "/home/deniz/Documents/mas_project/Qwen3-8B-quantized.w4a16"


@dataclass
class MASConfig:
    """Configuration for Multi-Agent System."""
    # Context limits (16K window)
    max_papers_context: int = 12000
    max_examples_context: int = 3000
    max_section_context: int = 4000
    
    # Critic thresholds - LOWERED for more iterations
    min_score_accept: int = 8  # Score >= 8/10 to accept (stricter)
    min_citation_ratio: float = 0.7  # At least 70% citations matched (stricter)
    max_revisions: int = 7  # Max revision loops (increased from 5)
    
    # Parallel writing
    enable_parallel: bool = True
    
    # Retrieval mode: "benchmark" (use provided papers) or "live" (Arxiv)
    retrieval_mode: str = "benchmark"
    
    # Theme/section limits
    min_themes: int = 2
    max_themes: int = 4
    
    # ROUGE optimization settings
    enable_rouge_refinement: bool = True  # Add ROUGE-boosting pass
    target_rouge1: float = 0.35  # Target ROUGE-1 score
    rouge_refinement_threshold: float = 0.25  # Refine if below this
    
    # Detailed logging
    enable_detailed_logging: bool = True


def get_llm(temperature: float = 0.6, thinking_mode: bool = True) -> ChatOpenAI:
    """
    Get configured LLM instance with Qwen3 optimal parameters.
    
    Qwen3 recommended settings (from HuggingFace):
    - Thinking mode: temperature=0.6, top_p=0.95, top_k=20
    - Non-thinking mode: temperature=0.7, top_p=0.8
    
    NOTE: vLLM with --enable-reasoning handles top_k automatically.
    """
    if thinking_mode:
        return ChatOpenAI(
            model=MODEL_NAME,
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            temperature=temperature,
            top_p=0.95
        )
    else:
        return ChatOpenAI(
            model=MODEL_NAME,
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            temperature=0.7,
            top_p=0.8
        )


def clean_output(content: str) -> str:
    """Clean LLM output by removing thinking blocks and artifacts."""
    # Remove <think>...</think> blocks
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Remove markdown code blocks
    content = content.replace("```", "").strip()
    
    # Remove common preambles
    preambles = [
        "Here is the ",
        "Here's the ",
        "Below is ",
        "The following is ",
    ]
    for p in preambles:
        if content.lower().startswith(p.lower()):
            # Find first period or newline after preamble
            idx = content.find('\n')
            if idx > 0 and idx < 100:
                content = content[idx:].strip()
    
    return content.strip()


def extract_thinking(content: str) -> str:
    """Extract the <think> block from LLM output for logging."""
    match = re.search(r'<think>(.*?)</think>', content, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_key_terms(text: str, top_n: int = 30) -> List[str]:
    """
    Extract key terms from reference text to guide ROUGE optimization.
    Returns most frequent meaningful words.
    """
    # Simple word frequency approach
    import re
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they',
        'their', 'which', 'where', 'when', 'what', 'there', 'these',
        'also', 'more', 'such', 'than', 'into', 'some', 'only', 'other',
        'over', 'most', 'very', 'each', 'both', 'between', 'after',
        'being', 'about', 'would', 'could', 'should', 'will', 'using'
    }
    
    # Count frequencies
    from collections import Counter
    word_counts = Counter(w for w in words if w not in stop_words)
    
    # Return top N terms
    return [word for word, count in word_counts.most_common(top_n)]


def format_paper_for_citation(title: str, authors: List[str], year: int, abstract: str) -> Dict:
    """
    Format paper info with (Author, Year) citation key.
    
    Returns dict with:
    - citation_key: "(Smith et al., 2020)"
    - full_text: Formatted paper info for context
    """
    # Create citation key
    if len(authors) == 0:
        author_cite = "Unknown"
    elif len(authors) == 1:
        # Get last name
        author_cite = authors[0].split()[-1] if authors[0] else "Unknown"
    elif len(authors) == 2:
        last1 = authors[0].split()[-1] if authors[0] else "Author1"
        last2 = authors[1].split()[-1] if authors[1] else "Author2"
        author_cite = f"{last1} and {last2}"
    else:
        author_cite = f"{authors[0].split()[-1]} et al." if authors[0] else "Unknown et al."
    
    citation_key = f"({author_cite}, {year})"
    
    # Full text for context
    full_text = f"""[{citation_key}]
Title: {title}
Authors: {', '.join(authors)}
Year: {year}
Abstract: {abstract}
"""
    
    return {
        "citation_key": citation_key,
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "full_text": full_text
    }

