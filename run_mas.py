"""
Multi-Agent System (MAS) - Interactive Runner
=============================================
Run the optimized Multi-Agent System on any topic or ArXiv ID.
This uses the same high-performance architecture as the benchmark.

Usage:
    python run_mas.py
"""

import os
import sys
import arxiv
import argparse
from typing import List, Dict, Tuple

# Import MAS components
from agents.base import MASConfig
from benchmark_optimized import OptimizedMASWorkflow, prepare_papers_with_consistent_citations
from utils.enhanced_logger import EnhancedMASLogger, set_logger

def retrieve_papers_from_arxiv(query: str, max_results: int = 10) -> List[str]:
    """Retrieve papers from ArXiv and format them for the MAS."""
    print(f"\n🔍 Searching ArXiv for: '{query}'...")
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers_data = []
    
    try:
        results = list(client.results(search))
        print(f"   Found {len(results)} papers.")
        
        for i, result in enumerate(results):
            # Format authors for citation
            authors = [a.name for a in result.authors]
            year = result.published.year
            
            # Create a rich text representation
            paper_info = (
                f"Title: {result.title}\n"
                f"Authors: {', '.join(authors)}\n"
                f"Year: {year}\n"
                f"Abstract: {result.summary}\n"
            )
            papers_data.append(paper_info)
            print(f"   [{i+1}] {result.title[:60]}... ({year})")
            
    except Exception as e:
        print(f"❌ Error fetching from ArXiv: {e}")
        return []
        
    return papers_data

def run_mas_interactive():
    print("\n" + "="*60)
    print("🤖 MULTI-AGENT SYSTEM (MAS) - Interactive Mode")
    print("="*60)
    
    # 1. Get User Input
    parser = argparse.ArgumentParser(description="Run MAS on a topic")
    parser.add_argument("topic", nargs="?", help="Research topic or ArXiv ID")
    args = parser.parse_args()

    topic = args.topic
    if not topic:
        topic = input("\nEnter research topic or ArXiv ID: ").strip()
    
    if not topic:
        print("Topic is required.")
        return

    # 2. Setup Logger
    logger = EnhancedMASLogger("mas_interactive")
    set_logger(logger)
    
    # 3. Retrieve Papers
    raw_papers = retrieve_papers_from_arxiv(topic, max_results=10)
    
    if not raw_papers:
        print("No papers found. Exiting.")
        return

    # 4. Process Papers for MAS (Standardize Citations)
    print("\n⚙️  Processing papers and standardizing citations...")
    papers_dict, expected_citations = prepare_papers_with_consistent_citations(raw_papers)
    
    # 5. Initialize MAS
    config = MASConfig()
    config.max_revisions = 2  # Optimized setting
    
    mas = OptimizedMASWorkflow(config)
    
    # 6. Run the Workflow
    print("\n🚀 Starting Multi-Agent Workflow...")
    print(f"   Agents: Planner -> Outliner -> Writer <-> Critic")
    
    try:
        generated_text, stats = mas.run(
            topic=topic,
            papers=papers_dict,
            target_abstract=f"Target Topic: {topic}", # Minimal context for ad-hoc
            fields_of_study="Computer Science"
        )
        
        # 7. Display Results
        print("\n" + "="*60)
        print("📄 GENERATED RELATED WORK")
        print("="*60 + "\n")
        print(generated_text)
        print("\n" + "="*60)
        
        # 8. Display Stats
        print("\n📊 SESSION STATS:")
        print(f"  • Time: {stats.get('total_duration', 'N/A')}")
        print(f"  • Themes Extracted: {stats.get('themes')}")
        print(f"  • Sections Written: {stats.get('sections')}")
        print(f"  • Revision Cycles: {stats.get('revision_cycles')}")
        print(f"  • Quality Scores (GRC):")
        print(f"    - Grounding: {stats.get('grounding', 0):.1f}/10")
        print(f"    - Coherence: {stats.get('coherence', 0):.1f}/10")
        print(f"    - Completeness: {stats.get('completeness', 0):.1f}/10")
        
        # Save to file
        filename = f"mas_output_{topic[:20].replace(' ', '_')}.txt"
        with open(filename, 'w') as f:
            f.write(f"TOPIC: {topic}\n\n")
            f.write(generated_text)
        print(f"\n💾 Output saved to: {filename}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_mas_interactive()
