"""
OARelatedWork Dataset Loader
https://huggingface.co/datasets/BUT-FIT/OARelatedWork

This dataset provides:
- Full related work sections as ground truth
- Full-text (or abstracts) of all cited papers
- Rich metadata including citation spans

We use the 'abstracts' configuration.
Optimized for 16K context window - can use more references and longer abstracts.
"""

# Configuration for context window
MAX_REFERENCES = 15  # Increased from 10 (16K context allows more)
MAX_ABSTRACT_CHARS = 2500  # Increased from 1500

from datasets import load_dataset
from typing import List, Dict, Any
import json

def extract_text_from_paragraphs(paragraphs: list) -> str:
    """
    Extracts plain text from the nested paragraph structure.
    Structure: list of paragraphs, each paragraph is list of sentences,
    each sentence is a dict with 'text' key.
    """
    text_parts = []
    if not paragraphs:
        return ""
    
    for para in paragraphs:
        if isinstance(para, list):
            for sent in para:
                if isinstance(sent, dict) and 'text' in sent:
                    text_parts.append(sent['text'])
                elif isinstance(sent, list):
                    # Deeper nesting
                    for item in sent:
                        if isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
        elif isinstance(para, dict) and 'text' in para:
            text_parts.append(para['text'])
    
    return " ".join(text_parts)

def extract_related_work(related_work_field: Any) -> str:
    """
    Extracts the related work text from the dataset's nested structure.
    The structure varies by configuration.
    """
    if isinstance(related_work_field, str):
        # Already plain text or JSON string
        try:
            data = json.loads(related_work_field)
            return extract_related_work(data)
        except:
            return related_work_field
    
    if isinstance(related_work_field, list):
        texts = []
        for section in related_work_field:
            if isinstance(section, dict):
                paragraphs = section.get('paragraphs', [])
                for para in paragraphs:
                    texts.append(extract_text_from_paragraphs(para))
            elif isinstance(section, list):
                texts.append(extract_text_from_paragraphs(section))
        return " ".join(texts)
    
    return str(related_work_field)

def extract_abstract_from_referenced(ref: dict) -> str:
    """
    Extracts the abstract text from a referenced paper.
    In the 'abstracts' config, the abstract is in the 'hierarchy' field.
    """
    hierarchy = ref.get('hierarchy', [])
    title = ref.get('title', 'Unknown Title')
    authors = ref.get('authors', [])
    year = ref.get('year', 'Unknown Year')
    
    # Extract abstract text from hierarchy
    abstract_text = ""
    if isinstance(hierarchy, list):
        for section in hierarchy:
            if isinstance(section, dict):
                title_path = section.get('title_path', [])
                # Look for abstract section
                if any('abstract' in str(t).lower() for t in title_path):
                    paragraphs = section.get('paragraphs', [])
                    abstract_text = extract_text_from_paragraphs(paragraphs)
                    break
        
        # If no explicit abstract, take first section
        if not abstract_text and hierarchy:
            first = hierarchy[0]
            if isinstance(first, dict):
                paragraphs = first.get('paragraphs', [])
                abstract_text = extract_text_from_paragraphs(paragraphs)
    
    # Format the reference
    author_str = ", ".join(authors[:3]) if authors else "Unknown"
    if len(authors) > 3:
        author_str += " et al."
    
    return f"Title: {title}\nAuthors: {author_str}\nYear: {year}\nAbstract: {abstract_text[:MAX_ABSTRACT_CHARS]}"

def extract_target_abstract(abstract_field: Any) -> str:
    """Extract abstract text from the target paper's abstract field."""
    if not abstract_field:
        return ""
    
    texts = []
    if isinstance(abstract_field, list):
        for para in abstract_field:
            if isinstance(para, list):
                for sent in para:
                    if isinstance(sent, dict) and 'text' in sent:
                        texts.append(sent['text'])
            elif isinstance(para, dict) and 'text' in para:
                texts.append(para['text'])
    
    return " ".join(texts)

def load_oarelatedwork_benchmark(num_samples: int = 10, split: str = "test") -> List[Dict]:
    """
    Loads samples from the OARelatedWork dataset.
    
    Now includes:
    - Target paper's abstract (context for what the paper is about)
    - Fields of study (thematic hints)
    - More referenced papers (16K context)
    
    Args:
        num_samples: Number of samples to load
        split: Dataset split ('train', 'validation', 'test')
    
    Returns:
        List of dicts with rich context for generation
    """
    print(f"Loading OARelatedWork dataset ({split} split)...")
    
    try:
        # Load the 'abstracts' configuration for efficiency
        dataset = load_dataset("BUT-FIT/OARelatedWork", "abstracts", split=split, streaming=True)
        
        samples = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            
            # Extract topic (title of the target paper)
            topic = item.get('title', f"Paper_{item.get('id', i)}")
            
            # NEW: Extract target paper's abstract (tells agent what the paper is about)
            target_abstract = extract_target_abstract(item.get('abstract', []))
            
            # NEW: Extract fields of study (helps with thematic grouping)
            fields_of_study = item.get('fields_of_study', [])
            fields_str = ", ".join(fields_of_study[:5]) if fields_of_study else "Not specified"
            
            # Extract source papers (referenced documents)
            referenced = item.get('referenced', [])
            source_papers = []
            for j, ref in enumerate(referenced[:MAX_REFERENCES]):  # 16K context allows more refs
                ref_text = extract_abstract_from_referenced(ref)
                # Add citation ID for our benchmark format
                source_papers.append(f"Citation ID: @cite_{j}\n{ref_text}")
            
            # Extract ground truth related work
            related_work = item.get('related_work', '')
            reference_text = extract_related_work(related_work)
            
            if source_papers and reference_text:
                samples.append({
                    'id': str(item.get('id', i)),
                    'topic': topic,
                    'target_abstract': target_abstract,  # NEW
                    'fields_of_study': fields_str,  # NEW
                    'source_papers': source_papers,
                    'reference_text': reference_text
                })
                print(f"  Loaded sample {len(samples)}: {topic[:50]}... ({len(source_papers)} refs)")
        
        print(f"Successfully loaded {len(samples)} samples from OARelatedWork")
        return samples
    
    except Exception as e:
        print(f"Error loading OARelatedWork: {e}")
        print("Attempting to install datasets library...")
        import subprocess
        subprocess.run(["pip", "install", "datasets", "-q"])
        return []

def preview_sample(sample: dict):
    """Pretty prints a sample for debugging."""
    print("=" * 80)
    print(f"ID: {sample['id']}")
    print(f"Topic: {sample['topic']}")
    print(f"Number of source papers: {len(sample['source_papers'])}")
    print("-" * 40)
    print("First source paper:")
    print(sample['source_papers'][0][:500] + "...")
    print("-" * 40)
    print("Reference (ground truth) Related Work:")
    print(sample['reference_text'][:1000] + "...")
    print("=" * 80)

if __name__ == "__main__":
    # Test the loader
    samples = load_oarelatedwork_benchmark(num_samples=3, split="test")
    if samples:
        preview_sample(samples[0])

