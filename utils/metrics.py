"""
Comprehensive Metrics for Related Work Generation.

Metrics Categories:
1. Lexical Overlap: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR, CHRF
2. Semantic Similarity: BERTScore
3. Citation Quality: Citation F1, Recall, Precision, Density
4. Content Quality: Abstractiveness, Length Ratio, Vocabulary Diversity
5. Coherence: Adjacent sentence coherence
6. Perplexity: Language model perplexity (lower = more fluent)
"""

import re
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np

# Core metrics
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# Additional NLG metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    import nltk
    # Ensure required NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. BLEU and METEOR will be skipped.")

# CHRF score
try:
    from sacrebleu.metrics import CHRF
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. CHRF will be skipped.")

# LSA (Latent Semantic Analysis) for topic-level similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. LSA will be skipped.")


@dataclass
class MetricsResult:
    """Complete metrics result for a single sample."""
    
    # Lexical (ROUGE)
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    
    # Classic NLG Metrics
    bleu: float = 0.0  # BLEU score (0-1)
    bleu_1: float = 0.0  # BLEU-1 (unigram)
    bleu_2: float = 0.0  # BLEU-2 (bigram)
    bleu_3: float = 0.0  # BLEU-3 (trigram)
    bleu_4: float = 0.0  # BLEU-4 (4-gram)
    meteor: float = 0.0  # METEOR score
    chrf: float = 0.0  # Character n-gram F-score
    
    # Semantic Similarity
    bert_score: float = 0.0  # Neural semantic similarity
    lsa_similarity: float = 0.0  # LSA topic-level similarity (replaces BLEU)
    
    # Citation Quality
    citation_f1: float = 0.0
    citation_recall: float = 0.0
    citation_precision: float = 0.0
    citation_density_generated: float = 0.0  # Citations per 100 words
    citation_density_reference: float = 0.0
    citation_density_ratio: float = 0.0  # How close to reference
    
    # Content Quality
    abstractiveness: float = 0.0  # % of novel n-grams
    length_ratio: float = 0.0  # Generated / Reference length
    vocabulary_diversity: float = 0.0  # Type-Token Ratio
    
    # Coherence
    coherence_score: float = 0.0  # Adjacent sentence similarity
    
    # Fluency
    perplexity: float = 0.0  # Language model perplexity (lower = better)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def summary(self) -> str:
        """Pretty print summary."""
        return (
            f"ROUGE: R1={self.rouge1:.3f} R2={self.rouge2:.3f} RL={self.rougeL:.3f}\n"
            f"BLEU: {self.bleu:.3f} (B1={self.bleu_1:.3f} B2={self.bleu_2:.3f} B3={self.bleu_3:.3f} B4={self.bleu_4:.3f})\n"
            f"METEOR: {self.meteor:.3f} | CHRF: {self.chrf:.3f}\n"
            f"BERTScore: {self.bert_score:.3f}\n"
            f"Citation: F1={self.citation_f1:.3f} R={self.citation_recall:.3f} P={self.citation_precision:.3f}\n"
            f"Content: Abstract={self.abstractiveness:.3f} LenRatio={self.length_ratio:.2f} Vocab={self.vocabulary_diversity:.3f}\n"
            f"Coherence: {self.coherence_score:.3f} | Perplexity: {self.perplexity:.1f}"
        )


def extract_citations(text: str) -> List[str]:
    """
    Extract all citations from text - both parenthetical and inline styles.
    
    Handles formats:
    - Parenthetical: (Smith, 2020), (Smith et al., 2020)
    - Inline: Smith (2020), Smith et al. (2020)
    - Names with special characters like accents
    """
    citations = set()
    
    # Pattern for parenthetical citations: (Author, Year) or (Author et al., Year)
    parenthetical_patterns = [
        r'\([A-Za-z][A-Za-z\u00C0-\u024F\'\-]+(?:\s+(?:and|&)\s+[A-Za-z][A-Za-z\u00C0-\u024F\'\-]+)*,?\s*\d{4}[a-z]?\)',
        r'\([A-Za-z][A-Za-z\u00C0-\u024F\'\-]+\s+et\s+al\.?,?\s*\d{4}[a-z]?\)',
        r'\([A-Za-z][A-Za-z\u00C0-\u024F\'\-]+\s+and\s+[A-Za-z][A-Za-z\u00C0-\u024F\'\-]+,?\s*\d{4}[a-z]?\)',
    ]
    
    for pattern in parenthetical_patterns:
        matches = re.findall(pattern, text)
        citations.update(matches)
    
    # Patterns for inline citations: Author (Year), Author et al. (Year)
    inline_patterns = [
        # Author (Year) - e.g., "Smith (2020)"
        r'([A-Za-z][A-Za-z\u00C0-\u024F\'\-]+)\s+\((\d{4}[a-z]?)\)',
        # Author et al. (Year) - e.g., "Smith et al. (2020)"
        r'([A-Za-z][A-Za-z\u00C0-\u024F\'\-]+)\s+et\s+al\.?\s+\((\d{4}[a-z]?)\)',
        # Author and Author (Year) - e.g., "Smith and Jones (2020)"
        r'([A-Za-z][A-Za-z\u00C0-\u024F\'\-]+)\s+and\s+([A-Za-z][A-Za-z\u00C0-\u024F\'\-]+)\s+\((\d{4}[a-z]?)\)',
    ]
    
    for pattern in inline_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) == 2:  # Single author
                author, year = match
                citations.add(f'({author}, {year})')
            elif len(match) == 3:  # Two authors
                # Check if it's et al pattern (match[1] might be 'al' part)
                if 'et al' in pattern:
                    author, year = match[0], match[1]
                    citations.add(f'({author} et al., {year})')
                else:
                    author1, author2, year = match
                    citations.add(f'({author1} and {author2}, {year})')
    
    # Special handling for et al. pattern (returns single capture group differently)
    et_al_pattern = r'([A-Za-z][A-Za-z\u00C0-\u024F\'\-]+)\s+et\s+al\.?\s+\((\d{4}[a-z]?)\)'
    et_al_matches = re.findall(et_al_pattern, text)
    for author, year in et_al_matches:
        citations.add(f'({author} et al., {year})')
    
    # Broad fallback for parenthetical citations
    broad_pattern = r'\([^()]+\d{4}[^()]*\)'
    broad_matches = re.findall(broad_pattern, text)
    for match in broad_matches:
        if re.search(r'[A-Za-z]{3,}', match) and re.search(r'\d{4}', match):
            if not any(skip in match.lower() for skip in ['figure', 'table', 'section', 'chapter', 'eq.', 'equation']):
                citations.add(match)
    
    return list(citations)


def compute_rouge(generated: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    if not generated or not reference:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def compute_bleu(generated: str, reference: str) -> Dict[str, float]:
    """
    Compute BLEU scores (1-4 gram and combined).
    
    BLEU measures n-gram precision with brevity penalty.
    Higher = better match with reference.
    """
    if not NLTK_AVAILABLE or not generated or not reference:
        return {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
    
    try:
        # Tokenize
        gen_tokens = word_tokenize(generated.lower())
        ref_tokens = word_tokenize(reference.lower())
        
        if not gen_tokens or not ref_tokens:
            return {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        
        # Use smoothing to handle zero counts
        smoother = SmoothingFunction()
        
        # Individual n-gram BLEU scores
        bleu_1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), 
                               smoothing_function=smoother.method1)
        bleu_2 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0),
                               smoothing_function=smoother.method1)
        bleu_3 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0),
                               smoothing_function=smoother.method1)
        bleu_4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                               smoothing_function=smoother.method1)
        
        # Combined BLEU (standard BLEU-4)
        bleu = bleu_4
        
        return {
            'bleu': bleu,
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4
        }
        
    except Exception as e:
        print(f"BLEU error: {e}")
        return {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}


def compute_meteor(generated: str, reference: str) -> float:
    """
    Compute METEOR score.
    
    METEOR considers synonyms, stemming, and word order.
    Often correlates better with human judgment than BLEU.
    Higher = better.
    """
    if not NLTK_AVAILABLE or not generated or not reference:
        return 0.0
    
    try:
        # Tokenize
        gen_tokens = word_tokenize(generated.lower())
        ref_tokens = word_tokenize(reference.lower())
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # METEOR expects reference as list of tokens, hypothesis as list of tokens
        score = meteor_score([ref_tokens], gen_tokens)
        return score
        
    except Exception as e:
        print(f"METEOR error: {e}")
        return 0.0


def compute_chrf(generated: str, reference: str) -> float:
    """
    Compute chrF score (Character n-gram F-score).
    
    chrF is robust to tokenization differences and morphological variations.
    Particularly useful for non-English or morphologically rich languages.
    Higher = better.
    """
    if not SACREBLEU_AVAILABLE or not generated or not reference:
        return 0.0
    
    try:
        chrf = CHRF()
        score = chrf.sentence_score(generated, [reference])
        return score.score / 100.0  # Normalize to 0-1
        
    except Exception as e:
        print(f"CHRF error: {e}")
        return 0.0


def compute_perplexity_simple(text: str) -> float:
    """
    Compute a simple perplexity proxy based on word frequency.
    
    Note: True perplexity requires a language model. This is a simplified
    version based on unigram probability from a reference corpus.
    
    Lower = more fluent/natural text.
    
    For true perplexity, you would need to use a model like GPT-2.
    """
    if not text:
        return float('inf')
    
    try:
        words = text.lower().split()
        if len(words) < 2:
            return float('inf')
        
        # Simple bigram-based entropy estimation
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)
        
        # Calculate log probability
        log_prob = 0.0
        for bigram in bigrams:
            # P(w2|w1) ≈ count(w1,w2) / count(w1)
            bigram_count = bigram_counts[bigram]
            unigram_count = unigram_counts[bigram[0]]
            
            if unigram_count > 0 and bigram_count > 0:
                prob = bigram_count / unigram_count
                log_prob += math.log2(prob)
            else:
                # Smoothing for unseen bigrams
                log_prob += math.log2(1e-10)
        
        # Perplexity = 2^(-1/N * sum(log2(P)))
        avg_log_prob = log_prob / len(bigrams)
        perplexity = math.pow(2, -avg_log_prob)
        
        # Cap at reasonable value
        return min(perplexity, 10000.0)
        
    except Exception as e:
        print(f"Perplexity error: {e}")
        return float('inf')


def compute_bert_score(generated: str, reference: str) -> float:
    """Compute BERTScore F1."""
    if not generated or not reference:
        return 0.0
    
    try:
        P, R, F1 = bert_score_fn([generated], [reference], lang="en", verbose=False)
        return float(F1[0])
    except Exception as e:
        print(f"BERTScore error: {e}")
        return 0.0


def compute_lsa_similarity(generated: str, reference: str, context_corpus: List[str] = None, n_components: int = 100) -> float:
    """
    Compute LSA (Latent Semantic Analysis) similarity.
    
    LSA captures topic-level similarity by:
    1. Converting texts to TF-IDF vectors
    2. Reducing dimensionality with SVD (finding latent topics)
    3. Computing cosine similarity in topic space
    
    This is better than BLEU for related work because:
    - Handles paraphrasing well
    - Captures conceptual overlap, not just n-grams
    - Works with different vocabulary expressing same ideas
    
    Args:
        generated: Generated text
        reference: Reference text
        context_corpus: Optional list of background texts (source papers) to help define the topic space
        n_components: Number of latent topics (default 100)
    
    Returns:
        LSA similarity score (0-1, higher = more similar topics)
    """
    if not SKLEARN_AVAILABLE:
        return 0.0
    
    if not generated or not reference:
        return 0.0
    
    try:
        # Combine texts for consistent vectorization
        # Use context corpus to build a better semantic space if available
        docs = [generated, reference]
        if context_corpus:
            docs.extend(context_corpus)
            
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        tfidf_matrix = vectorizer.fit_transform(docs)
        
        # Determine valid number of components
        # n_components must be < min(n_samples, n_features)
        n_samples, n_features = tfidf_matrix.shape
        max_components = min(n_samples, n_features) - 1
        
        # If we have too few documents for SVD (e.g. only 2), LSA is degenerate (always 1.0).
        # Fallback to direct Cosine Similarity on TF-IDF vectors in that case.
        if max_components < 2:
            # Fallback: TF-IDF Cosine Similarity (no SVD)
            # This happens if we don't have enough context papers
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(max(0.0, min(1.0, similarity)))
            
        n_comp = min(n_components, max_components)
        
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        lsa_matrix = svd.fit_transform(tfidf_matrix)
        
        # Cosine similarity between generated (index 0) and reference (index 1) in LSA space
        similarity = cosine_similarity(lsa_matrix[0:1], lsa_matrix[1:2])[0][0]
        
        return float(max(0.0, min(1.0, similarity)))  # Clamp to [0, 1]
        
    except Exception as e:
        print(f"LSA similarity error: {e}")
        return 0.0


def compute_citation_metrics(generated: str, reference: str, 
                            expected_citations: List[str] = None) -> Dict[str, float]:
    """
    Compute comprehensive citation metrics.
    
    Args:
        generated: Generated text
        reference: Reference text
        expected_citations: Optional list of expected citation keys
    
    Returns:
        Dict with citation_f1, citation_recall, citation_precision, densities
    """
    gen_citations = set(extract_citations(generated))
    ref_citations = set(extract_citations(reference))
    
    # Use expected citations if provided, otherwise use reference citations
    target_citations = set(expected_citations) if expected_citations else ref_citations
    
    # Handle edge cases
    if not target_citations and not gen_citations:
        return {
            'citation_f1': 1.0,
            'citation_recall': 1.0,
            'citation_precision': 1.0,
            'citation_density_generated': 0.0,
            'citation_density_reference': 0.0,
            'citation_density_ratio': 1.0
        }
    
    if not target_citations:
        return {
            'citation_f1': 0.0 if gen_citations else 1.0,
            'citation_recall': 1.0,
            'citation_precision': 0.0 if gen_citations else 1.0,
            'citation_density_generated': _compute_density(generated, gen_citations),
            'citation_density_reference': 0.0,
            'citation_density_ratio': 0.0
        }
    
    if not gen_citations:
        return {
            'citation_f1': 0.0,
            'citation_recall': 0.0,
            'citation_precision': 0.0,
            'citation_density_generated': 0.0,
            'citation_density_reference': _compute_density(reference, ref_citations),
            'citation_density_ratio': 0.0
        }
    
    # Compute precision, recall, F1
    # For matching, we use fuzzy matching (author + year)
    true_positives = 0
    for gen_cite in gen_citations:
        for target_cite in target_citations:
            if _citations_match(gen_cite, target_cite):
                true_positives += 1
                break
    
    precision = true_positives / len(gen_citations)
    recall = true_positives / len(target_citations)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Compute densities
    gen_density = _compute_density(generated, gen_citations)
    ref_density = _compute_density(reference, ref_citations)
    density_ratio = min(gen_density, ref_density) / max(gen_density, ref_density) if max(gen_density, ref_density) > 0 else 1.0
    
    return {
        'citation_f1': f1,
        'citation_recall': recall,
        'citation_precision': precision,
        'citation_density_generated': gen_density,
        'citation_density_reference': ref_density,
        'citation_density_ratio': density_ratio
    }


def _citations_match(cite1: str, cite2: str) -> bool:
    """Check if two citations refer to the same work (fuzzy matching)."""
    # Extract author and year
    def extract_parts(cite):
        # Remove parentheses
        cite = cite.strip('()')
        # Find year
        year_match = re.search(r'\d{4}', cite)
        year = year_match.group() if year_match else ''
        # Find author (first word that looks like a name - allows accented chars)
        author_match = re.search(r'[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F\'\-]+', cite)
        author = author_match.group() if author_match else ''
        return author.lower(), year
    
    a1, y1 = extract_parts(cite1)
    a2, y2 = extract_parts(cite2)
    
    # Exact match
    if a1 == a2 and y1 == y2:
        return True
    
    # Fuzzy match: check if one author name contains the other (handles partial matches)
    if y1 == y2 and (a1 in a2 or a2 in a1) and min(len(a1), len(a2)) >= 3:
        return True
    
    return False


def _compute_density(text: str, citations: set) -> float:
    """Compute citations per 100 words."""
    words = len(text.split())
    if words == 0:
        return 0.0
    return len(citations) * 100 / words


def compute_abstractiveness(generated: str, sources: List[str], n: int = 3) -> float:
    """
    Compute abstractiveness: percentage of novel n-grams.
    
    Higher = more original text (not copied from sources)
    Lower = more copied/extractive
    
    Args:
        generated: Generated text
        sources: List of source texts (abstracts)
        n: N-gram size (default 3)
    
    Returns:
        Percentage of n-grams in generated that don't appear in sources
    """
    def get_ngrams(text: str, n: int) -> set:
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    
    gen_ngrams = get_ngrams(generated, n)
    
    if not gen_ngrams:
        return 0.0
    
    # Combine all source n-grams
    source_ngrams = set()
    for source in sources:
        source_ngrams.update(get_ngrams(source, n))
    
    # Count novel n-grams
    novel = gen_ngrams - source_ngrams
    
    return len(novel) / len(gen_ngrams)


def compute_length_ratio(generated: str, reference: str) -> float:
    """
    Compute length ratio: generated / reference.
    
    Ideal is close to 1.0.
    """
    gen_len = len(generated.split())
    ref_len = len(reference.split())
    
    if ref_len == 0:
        return 0.0
    
    return gen_len / ref_len


def compute_vocabulary_diversity(text: str) -> float:
    """
    Compute Type-Token Ratio (TTR): unique words / total words.
    
    Higher = more diverse vocabulary
    """
    words = text.lower().split()
    
    if not words:
        return 0.0
    
    unique_words = set(words)
    return len(unique_words) / len(words)


def compute_coherence(text: str) -> float:
    """
    Compute simple coherence score based on adjacent sentence similarity.
    
    Uses word overlap as a proxy for coherence.
    More sophisticated methods could use sentence embeddings.
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 1.0  # Single sentence is coherent by default
    
    # Compute word overlap between adjacent sentences
    similarities = []
    for i in range(len(sentences) - 1):
        words1 = set(sentences[i].lower().split())
        words2 = set(sentences[i + 1].lower().split())
        
        if not words1 or not words2:
            continue
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0
        similarities.append(similarity)
    
    if not similarities:
        return 1.0
    
    return sum(similarities) / len(similarities)


def compute_all_metrics(generated: str, reference: str, 
                        source_papers: List[str] = None,
                        expected_citations: List[str] = None) -> MetricsResult:
    """
    Compute all metrics for a single sample.
    
    Args:
        generated: Generated Related Work text
        reference: Ground truth Related Work text
        source_papers: List of source paper abstracts (for abstractiveness)
        expected_citations: List of expected citation keys
    
    Returns:
        MetricsResult with all metrics
    """
    result = MetricsResult()
    
    # ROUGE
    rouge = compute_rouge(generated, reference)
    result.rouge1 = rouge['rouge1']
    result.rouge2 = rouge['rouge2']
    result.rougeL = rouge['rougeL']
    
    # BLEU
    bleu = compute_bleu(generated, reference)
    result.bleu = bleu['bleu']
    result.bleu_1 = bleu['bleu_1']
    result.bleu_2 = bleu['bleu_2']
    result.bleu_3 = bleu['bleu_3']
    result.bleu_4 = bleu['bleu_4']
    
    # METEOR
    result.meteor = compute_meteor(generated, reference)
    
    # CHRF
    result.chrf = compute_chrf(generated, reference)
    
    # BERTScore (neural semantic similarity)
    result.bert_score = compute_bert_score(generated, reference)
    
    # LSA Similarity (topic-level semantic similarity - better than BLEU!)
    # Pass source papers as background corpus for better topic modeling
    result.lsa_similarity = compute_lsa_similarity(generated, reference, context_corpus=source_papers)
    
    # Citation metrics
    citation_metrics = compute_citation_metrics(generated, reference, expected_citations)
    result.citation_f1 = citation_metrics['citation_f1']
    result.citation_recall = citation_metrics['citation_recall']
    result.citation_precision = citation_metrics['citation_precision']
    result.citation_density_generated = citation_metrics['citation_density_generated']
    result.citation_density_reference = citation_metrics['citation_density_reference']
    result.citation_density_ratio = citation_metrics['citation_density_ratio']
    
    # Content quality
    if source_papers:
        result.abstractiveness = compute_abstractiveness(generated, source_papers)
    
    result.length_ratio = compute_length_ratio(generated, reference)
    result.vocabulary_diversity = compute_vocabulary_diversity(generated)
    
    # Coherence
    result.coherence_score = compute_coherence(generated)
    
    # Perplexity (simple estimation)
    result.perplexity = compute_perplexity_simple(generated)
    
    return result


def aggregate_metrics(results: List[MetricsResult]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple samples.
    
    Returns mean and std for each metric.
    """
    if not results:
        return {}
    
    # Collect all metric values
    metrics = {}
    for field in MetricsResult.__dataclass_fields__:
        values = [getattr(r, field) for r in results]
        metrics[f"{field}_mean"] = np.mean(values)
        metrics[f"{field}_std"] = np.std(values)
    
    return metrics


# --- Standalone test ---
if __name__ == "__main__":
    print("Testing Metrics Module...")
    print(f"NLTK available: {NLTK_AVAILABLE}")
    print(f"SacreBLEU available: {SACREBLEU_AVAILABLE}")
    
    generated = """
    The transformer architecture introduced by (Vaswani et al., 2017) revolutionized 
    natural language processing by eliminating recurrence. Building on attention mechanisms 
    proposed by (Bahdanau et al., 2015), transformers enable parallel processing of sequences.
    Recent work by (Devlin et al., 2019) demonstrated the effectiveness of pre-training.
    """
    
    reference = """
    Attention mechanisms have become fundamental to sequence modeling. The seminal work 
    by (Bahdanau et al., 2015) introduced attention for neural machine translation. 
    This was later extended by (Vaswani et al., 2017) who proposed the Transformer, 
    using self-attention exclusively. BERT (Devlin et al., 2019) showed impressive results.
    """
    
    sources = [
        "We propose a new attention mechanism for neural machine translation...",
        "The Transformer model uses self-attention to compute representations...",
        "BERT is a language model that learns bidirectional representations..."
    ]
    
    print("\nComputing metrics...")
    metrics = compute_all_metrics(generated, reference, sources)
    
    print("\n" + "="*50)
    print("METRICS RESULTS")
    print("="*50)
    print(metrics.summary())
    
    print("\n" + "-"*50)
    print("All metrics (detailed):")
    print("-"*50)
    for k, v in metrics.to_dict().items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.4f}")
        else:
            print(f"  {k:<30}: {v}")

