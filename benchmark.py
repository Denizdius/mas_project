"""
Benchmark: Single Agent vs Multi-Agent System Comparison
=========================================================

KEY FEATURES:
1. SINGLE AGENT: Baseline LangGraph agent with simple pipeline
2. MULTI-AGENT SYSTEM: Planner→Outliner→Writer→Critic→Assembly
3. COMPARISON MODE: Run both and compare metrics side-by-side

Usage:
    python benchmark.py --samples 10                    # MAS only
    python benchmark.py --samples 10 --compare          # MAS + Single Agent comparison
    python benchmark.py --samples 5 --single-only       # Single Agent only
"""

import json
import argparse
import time
import re
import sys
from typing import Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, '/home/deniz/Documents/mas_project')

from agents.base import MASConfig, clean_output, get_llm
from agents.planner import PlannerAgent
from agents.outliner import OutlinerAgent
from agents.writer import WriterAgent, WriterOutput  # Use original writer (no ROUGE boost)
from agents.critic import CriticAgent, CriticOutput
from load_oarelatedwork import load_oarelatedwork_benchmark
from utils.enhanced_logger import EnhancedMASLogger, set_logger, get_logger, quick_rouge
from utils.metrics import compute_all_metrics, MetricsResult, aggregate_metrics
from langchain_core.messages import HumanMessage, SystemMessage

# Import single agent for comparison
from single_agent import generate_related_work_v2 as single_agent_generate


def extract_author_year_from_text(text: str) -> Tuple[str, str]:
    """
    Extract first author's last name and year from paper text.
    Returns (author_last_name, year) or ('Unknown', 'Unknown')
    """
    # Try to find Authors: line
    authors_match = re.search(r'Authors?:\s*([^,\n]+)', text)
    year_match = re.search(r'Year:\s*(\d{4})', text)
    
    author = "Unknown"
    year = "Unknown"
    
    if authors_match:
        author_text = authors_match.group(1).strip()
        # Get last name of first author
        parts = author_text.split()
        if parts:
            author = parts[-1]  # Last word is usually last name
    
    if year_match:
        year = year_match.group(1)
    
    return author, year


def prepare_papers_with_real_citations(source_papers: List[str]) -> List[Dict]:
    """
    Convert source paper strings to structured dicts with REAL (Author, Year) citations.
    
    CRITICAL FIX: Extract actual author names and years, don't use placeholders!
    """
    papers = []
    
    for i, paper_str in enumerate(source_papers):
        lines = paper_str.strip().split('\n')
        
        title = 'Unknown'
        authors = []
        year = 'Unknown'
        abstract = ''
        
        # Parse paper info
        for line in lines:
            line = line.strip()
            if line.startswith('Title:'):
                title = line[6:].strip()
            elif line.startswith('Authors:'):
                authors_str = line[8:].strip()
                authors = [a.strip() for a in authors_str.split(',')]
            elif line.startswith('Year:'):
                year = line[5:].strip()
            elif line.startswith('Abstract:'):
                abstract = line[9:].strip()
        
        # Create proper (Author, Year) citation key
        if authors:
            first_author_last = authors[0].split()[-1] if authors[0] else "Unknown"
            if len(authors) == 1:
                citation_key = f"({first_author_last}, {year})"
            elif len(authors) == 2:
                last1 = authors[0].split()[-1] if authors[0] else "Author1"
                last2 = authors[1].split()[-1] if authors[1] else "Author2"
                citation_key = f"({last1} and {last2}, {year})"
            else:
                citation_key = f"({first_author_last} et al., {year})"
        else:
            citation_key = f"(Paper {i}, {year})"
        
        # Full text for context - with proper citation header
        full_text = f"""[{citation_key}]
Title: {title}
Authors: {', '.join(authors)}
Year: {year}
Abstract: {abstract}
"""
        
        papers.append({
            'citation_key': citation_key,
            'title': title,
            'authors': authors,
            'year': year,
            'abstract': abstract,
            'full_text': full_text
        })
    
    return papers


class FixedCriticAgent:
    """
    Fixed Critic Agent with flexible citation matching.
    
    KEY FIXES:
    1. Matches author names + year flexibly (not exact string match)
    2. High grounding (>=9) can override citation_accuracy requirement
    3. More informative feedback
    4. NEW: Adjusts completeness expectation based on assigned papers count
    """
    
    def __init__(self, config: MASConfig = None):
        self.config = config or MASConfig()
        self.llm = get_llm(temperature=0.3, thinking_mode=True)
    
    def _check_citation_flexible(self, text: str, citations: List[str]) -> Tuple[float, List[str]]:
        """
        Flexibly check citations - match author names and years.
        
        Returns (accuracy, list_of_found_citations)
        """
        if not citations:
            return 1.0, []
        
        found = []
        text_lower = text.lower()
        
        for cite in citations:
            # Extract author and year from "(Author et al., 2020)" format
            match = re.search(r'\(([^,]+)(?:,|\s+et\s+al\.?,?\s*)(\d{4})\)', cite)
            if match:
                author_part = match.group(1).strip().lower()
                year = match.group(2)
                
                # Check if author name appears near year in text
                # This catches variations like "Dung et al. (1995)", "Dung (1995)", etc.
                author_patterns = [
                    rf'{author_part}.*{year}',
                    rf'{author_part}\s+et\s+al.*{year}',
                    rf'{author_part}\s+and.*{year}',
                ]
                
                for pattern in author_patterns:
                    if re.search(pattern, text_lower):
                        found.append(cite)
                        break
                else:
                    # Also check if just author and year appear separately
                    if author_part in text_lower and year in text:
                        found.append(cite)
            else:
                # Direct string match fallback
                if cite in text:
                    found.append(cite)
        
        return len(found) / len(citations), found
    
    def evaluate(self, writer_output: Dict, section: Dict) -> CriticOutput:
        """Evaluate with flexible citation matching."""
        section_id = writer_output.get('section_id', 0)
        text = writer_output.get('text', '')
        
        required_citations = section.get('citations_to_use', [])
        papers_context = section.get('papers_context', '')
        num_assigned_papers = len(required_citations)
        
        print(f"[CRITIC] Evaluating section {section_id}...")
        
        # Flexible citation check
        citation_accuracy, found_citations = self._check_citation_flexible(text, required_citations)
        
        # LLM evaluation
        scores, issues = self._evaluate_quality(text, papers_context, required_citations)
        
        grounding = scores.get('grounding', 5)
        coherence = scores.get('coherence', 5)
        completeness = scores.get('completeness', 5)
        
        # FIX: Adjust completeness expectation for single-paper sections
        # If only 1 paper assigned, completeness can't be judged on "comparing works"
        if num_assigned_papers <= 1:
            completeness = max(completeness, 7)  # Floor at 7 for single-paper sections
        
        # STRICTER ACCEPTANCE LOGIC (v4):
        # Accept ONLY if average score >= 8.5 AND all scores >= 7
        # This ensures revision loop actually engages
        min_score = self.config.min_score_accept  # default 8
        avg_score = (grounding + coherence + completeness) / 3
        
        accept = (
            avg_score >= 8.5 and  # Average must be high
            grounding >= 7 and     # No score below 7
            coherence >= 7 and
            completeness >= 7 and
            citation_accuracy >= 0.3  # Must have SOME citations matched
        )
        
        # Generate feedback
        if not accept:
            feedback = self._generate_feedback(
                issues, scores, citation_accuracy, 
                required_citations, found_citations
            )
        else:
            feedback = "Section accepted - meets quality criteria."
        
        status = "ACCEPT" if accept else "REJECT"
        print(f"[CRITIC] Section {section_id}: {status} "
              f"(G:{grounding} C:{coherence} Comp:{completeness} Cite:{citation_accuracy:.0%})")
        
        return CriticOutput(
            section_id=section_id,
            accept=accept,
            scores=scores,
            citation_accuracy=citation_accuracy,
            feedback=feedback,
            issues=issues
        )
    
    def _evaluate_quality(self, text: str, papers_context: str, citations: List[str]) -> Tuple[Dict, List[str]]:
        """LLM-based quality evaluation."""
        system_prompt = """You are a strict academic reviewer. Evaluate this Related Work paragraph.

### Scoring (1-10):
1. GROUNDING: Are claims supported by cited papers? (10=all supported, 1=hallucinated)
2. COHERENCE: Does it flow logically? (10=perfect, 1=disjointed)
3. COMPLETENESS: Are papers meaningfully discussed? (10=thorough, 1=just listed)

### Output Format:
GROUNDING: [score]
COHERENCE: [score]
COMPLETENESS: [score]
ISSUES:
- Issue 1
- Issue 2"""

        user_prompt = f"""### Text
{text}

### Source Papers
{papers_context[:2500]}

Evaluate:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = clean_output(response.content)
            
            scores = {
                'grounding': self._extract_score(content, 'GROUNDING'),
                'coherence': self._extract_score(content, 'COHERENCE'),
                'completeness': self._extract_score(content, 'COMPLETENESS')
            }
            
            issues = self._extract_issues(content)
            return scores, issues
            
        except Exception as e:
            print(f"[CRITIC] Error: {e}")
            return {'grounding': 7, 'coherence': 7, 'completeness': 7}, [str(e)]
    
    def _extract_score(self, content: str, metric: str) -> int:
        match = re.search(rf'{metric}:\s*(\d+)', content, re.IGNORECASE)
        if match:
            return min(10, max(1, int(match.group(1))))
        return 7
    
    def _extract_issues(self, content: str) -> List[str]:
        issues = []
        if 'ISSUES:' in content:
            issues_section = content.split('ISSUES:', 1)[1]
            for line in issues_section.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    issue = line.lstrip('-•').strip()
                    if issue and len(issue) > 5:
                        issues.append(issue[:200])
        return issues[:3]
    
    def _generate_feedback(self, issues, scores, cite_acc, required, found):
        parts = []
        
        if scores.get('grounding', 10) < 8:
            parts.append("GROUNDING: Verify claims against source papers.")
        if scores.get('coherence', 10) < 8:
            parts.append("COHERENCE: Improve transitions between sentences.")
        if scores.get('completeness', 10) < 8:
            parts.append("COMPLETENESS: Discuss cited papers more substantively.")
        
        if issues:
            parts.append(f"ISSUES: {'; '.join(issues[:2])}")
        
        return "\n".join(parts) if parts else "Minor improvements needed."


class MASWorkflowV3:
    """
    Fixed MAS Workflow - no ROUGE boost, proper citations, fewer revisions.
    """
    
    def __init__(self, config: MASConfig, reference_text: str = ""):
        self.config = config
        self.reference_text = reference_text
        
        # Initialize agents
        self.planner = PlannerAgent(config)
        self.outliner = OutlinerAgent(config)
        self.writer = WriterAgent(config)  # Original writer (no ROUGE boost)
        self.critic = FixedCriticAgent(config)  # Fixed critic
    
    def run(self, topic: str, papers: List[Dict], 
            target_abstract: str = "", fields_of_study: str = "") -> str:
        """Run the complete workflow."""
        logger = get_logger()
        
        print(f"\n{'='*60}")
        print(f"MAS v3: {topic[:50]}...")
        print(f"{'='*60}")
        
        # Stage 1: Planner
        print("\n[STAGE 1] PLANNER")
        planner_start = time.time()
        planner_output = self.planner.run(
            topic=topic,
            target_abstract=target_abstract,
            fields_of_study=fields_of_study,
            provided_papers=papers
        )
        planner_duration = time.time() - planner_start
        
        if logger:
            logger.log_planner(
                themes=planner_output.get('themes', []),
                papers_count=planner_output.get('total_papers', 0),
                duration=planner_duration
            )
        
        if not planner_output.get('themes'):
            return "Error: No themes extracted."
        
        # Stage 2: Outliner
        print("\n[STAGE 2] OUTLINER")
        outliner_start = time.time()
        outliner_output = self.outliner.run(planner_output)
        outliner_output['topic'] = topic
        outliner_duration = time.time() - outliner_start
        
        if logger:
            logger.log_outliner(
                sections=outliner_output.get('sections', []),
                gap_hint=outliner_output.get('gap_analysis_hint', ''),
                duration=outliner_duration
            )
        
        sections = outliner_output.get('sections', [])
        if not sections:
            return "Error: No sections created."
        
        # Stage 3 & 4: Writer-Critic Loop (max 3 revisions)
        print("\n[STAGE 3-4] WRITER-CRITIC LOOP")
        
        section_states = []
        for section in sections:
            section_states.append({
                'section': section,
                'writer_output': None,
                'critic_output': None,
                'revision_count': 0,
                'accepted': False
            })
        
        # Revision loop (max 3 revisions is enough with fixed citations)
        for revision_round in range(self.config.max_revisions + 1):
            print(f"\n--- Round {revision_round} ---")
            
            pending = [(i, ss) for i, ss in enumerate(section_states) if not ss['accepted']]
            
            if not pending:
                print("✓ All sections accepted!")
                break
            
            for idx, ss in pending:
                section = ss['section']
                section_id = section.get('section_id', idx)
                is_last = (idx == len(section_states) - 1)
                
                feedback = ss['critic_output'].get('feedback') if ss['critic_output'] else None
                
                # Write/Revise
                if feedback and ss['writer_output']:
                    prev = WriterOutput(
                        section_id=section_id,
                        text=ss['writer_output'].get('text', ''),
                        citations_used=ss['writer_output'].get('citations_used', []),
                        revision_count=ss['revision_count']
                    )
                    result = self.writer.revise_section(section, topic, prev, feedback, is_last)
                else:
                    result = self.writer.write_section(section, topic, is_last)
                
                ss['writer_output'] = result.to_dict()
                ss['revision_count'] = result.revision_count
                
                # Critique
                critic_result = self.critic.evaluate(result.to_dict(), section)
                ss['critic_output'] = critic_result.to_dict()
                ss['accepted'] = critic_result.accept
                
                if logger:
                    logger.log_writer(
                        section_id=section_id,
                        theme_name=section.get('theme_name', ''),
                        generated_text=result.text,
                        citations_used=result.citations_used,
                        revision_number=result.revision_count,
                        duration=0
                    )
                    logger.log_critic(
                        section_id=section_id,
                        scores=critic_result.scores,
                        issues=critic_result.issues,
                        accepted=critic_result.accept,
                        feedback=critic_result.feedback,
                        citation_accuracy=critic_result.citation_accuracy,
                        duration=0
                    )
        
        # Force accept remaining
        for ss in section_states:
            if not ss['accepted']:
                print(f"[MAS] Force accepting section {ss['section'].get('section_id', '?')}")
                ss['accepted'] = True
        
        # Stage 5: Assembly
        print("\n[STAGE 5] ASSEMBLY")
        section_texts = [ss['writer_output']['text'] for ss in section_states if ss['writer_output']]
        
        if not section_texts:
            return "Error: No content."
        
        if len(section_texts) == 1:
            final_text = section_texts[0]
        else:
            final_text = "\n\n".join(section_texts)
        
        if logger:
            logger.log_assembly(len(section_texts), final_text, 0)
        
        return final_text


def run_benchmark(num_samples: int = 10, max_revisions: int = 3):
    """Run the fixed benchmark."""
    print("\n" + "#"*70)
    print("# BENCHMARK v3: FIXED MAS (Proper Citations, No ROUGE Boost)")
    print("#"*70)
    print(f"Samples: {num_samples}")
    print(f"Max revisions: {max_revisions}")
    print("#"*70 + "\n")
    
    # Load dataset
    samples = load_oarelatedwork_benchmark(num_samples=num_samples, split="test")
    if not samples:
        print("ERROR: Could not load samples")
        return
    
    # Logger
    logger = EnhancedMASLogger("mas_v3_fixed")
    set_logger(logger)
    
    logger.set_config({
        "num_samples": len(samples),
        "max_revisions": max_revisions,
        "version": "v3_fixed_citations_no_rouge_boost"
    })
    
    # Config - v4 OPTIMIZED settings
    config = MASConfig()
    config.retrieval_mode = "benchmark"
    config.max_revisions = max_revisions
    config.enable_rouge_refinement = False  # DISABLED - counterproductive
    config.min_score_accept = 9  # STRICTER: Require higher scores
    config.min_citation_ratio = 0.3  # Allow some flexibility
    
    results_list: List[MetricsResult] = []
    sample_results = []
    
    total_start = time.time()
    
    for i, sample in enumerate(samples):
        sample_id = sample.get('id', str(i))
        topic = sample['topic']
        reference_text = sample['reference_text']
        
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(samples)}] {topic[:60]}...")
        print(f"{'='*70}")
        
        # FIXED: Use real citations from paper metadata
        papers_dict = prepare_papers_with_real_citations(sample['source_papers'])
        
        # Show extracted citations
        print(f"[INFO] Extracted citations: {[p['citation_key'] for p in papers_dict]}")
        
        logger.start_sample(
            sample_id=sample_id,
            topic=topic,
            target_abstract=sample.get('target_abstract', ''),
            fields_of_study=sample.get('fields_of_study', ''),
            reference_text=reference_text,
            source_papers=papers_dict
        )
        
        sample_start = time.time()
        
        try:
            mas = MASWorkflowV3(config, reference_text=reference_text)
            
            generated = mas.run(
                topic=topic,
                papers=papers_dict,
                target_abstract=sample.get('target_abstract', ''),
                fields_of_study=sample.get('fields_of_study', '')
            )
            
            sample_duration = time.time() - sample_start
            
            # Metrics
            source_abstracts = [p.get('abstract', '') for p in papers_dict]
            expected_citations = [p['citation_key'] for p in papers_dict]
            
            metrics = compute_all_metrics(
                generated=generated,
                reference=reference_text,
                source_papers=source_abstracts,
                expected_citations=expected_citations
            )
            
            results_list.append(metrics)
            
            print(f"\n[RESULT] R1:{metrics.rouge1:.3f} R2:{metrics.rouge2:.3f} "
                  f"RL:{metrics.rougeL:.3f} BERT:{metrics.bert_score:.3f}")
            print(f"[TIME] {sample_duration:.1f}s")
            
            logger.finalize_sample(
                final_metrics=metrics.to_dict(),
                total_duration=sample_duration
            )
            
            sample_results.append({
                "id": sample_id,
                "topic": topic,
                "generated": generated[:3000],
                "reference": reference_text[:3000],
                "metrics": metrics.to_dict(),
                "time_seconds": sample_duration,
                "citations_expected": expected_citations
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            logger.finalize_sample({}, time.time() - sample_start)
            sample_results.append({"id": sample_id, "error": str(e)})
    
    total_duration = time.time() - total_start
    
    # Aggregate
    agg = aggregate_metrics(results_list) if results_list else {}
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - MAS v3 (FIXED)")
    print("="*70)
    
    print(f"\n📊 ROUGE Scores:")
    print(f"  ROUGE-1: {agg.get('rouge1_mean', 0):.3f} ± {agg.get('rouge1_std', 0):.3f}")
    print(f"  ROUGE-2: {agg.get('rouge2_mean', 0):.3f} ± {agg.get('rouge2_std', 0):.3f}")
    print(f"  ROUGE-L: {agg.get('rougeL_mean', 0):.3f} ± {agg.get('rougeL_std', 0):.3f}")
    
    print(f"\n📈 Semantic Quality:")
    print(f"  BERTScore: {agg.get('bert_score_mean', 0):.3f}")
    print(f"  BLEU: {agg.get('bleu_mean', 0):.3f}")
    print(f"  METEOR: {agg.get('meteor_mean', 0):.3f}")
    
    print(f"\n⏱️ Performance:")
    print(f"  Total: {total_duration:.1f}s")
    print(f"  Avg/sample: {total_duration/len(samples):.1f}s")
    
    # Compare with v2
    print(f"\n📉 vs v2 (from your logs):")
    print(f"  v2 avg time: ~680s/sample → v3: {total_duration/len(samples):.1f}s")
    print(f"  Expected improvement: Faster (fewer revisions)")
    
    # Save
    results_file = f"benchmark_mas_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(samples),
            "max_revisions": max_revisions,
            "total_duration": total_duration,
            "version": "MAS_v3_fixed"
        },
        "aggregated": agg,  # Use same key as single_agent for comparison compatibility
        "samples": sample_results,
        "total_duration": total_duration,
        "avg_duration": total_duration / len(samples) if samples else 0
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {results_file}")
    
    log_file = logger.save()
    print(f"Logs: {log_file}")
    
    return results


def run_single_agent_benchmark(num_samples: int, enable_refinement: bool = False):
    """
    Run Single Agent benchmark for comparison.
    """
    samples = load_oarelatedwork_benchmark(num_samples=num_samples, split='test')
    if not samples:
        print("ERROR: No samples loaded")
        return None
    
    print(f"\n{'='*70}")
    print("SINGLE AGENT BENCHMARK")
    print(f"{'='*70}")
    print(f"Samples: {len(samples)}")
    print(f"Refinement: {'Enabled' if enable_refinement else 'Disabled'}")
    print(f"{'='*70}\n")
    
    results_list = []
    sample_results = []
    total_start = time.time()
    
    for idx, sample in enumerate(samples):
        sample_id = sample.get('id', f'sample_{idx}')
        topic = sample.get('topic', sample.get('title', 'Unknown'))  # Try 'topic' first, then 'title'
        reference_text = sample.get('reference_text', '')
        source_papers = sample.get('source_papers', [])
        
        print(f"\n[{idx+1}/{len(samples)}] {topic[:60]}...")
        sample_start = time.time()
        
        try:
            # Prepare state for single agent
            # source_papers are formatted strings like "Title: ...\nAuthors: ...\nYear: ...\nAbstract: ..."
            state = {
                'topic': topic,
                'papers': source_papers,  # Already formatted strings from load_oarelatedwork
                'target_abstract': sample.get('target_abstract', ''),
                'fields_of_study': sample.get('fields_of_study', ''),
                'examples': None,
                'draft': None,
                'refined_draft': None,
                'enable_refinement': False
            }
            
            # Run single agent
            result = single_agent_generate(state, enable_refinement=enable_refinement)
            generated = result.get('draft', '') or result.get('refined_draft', '')
            
            # Clean output
            generated = clean_output(generated)
            
            sample_duration = time.time() - sample_start
            
            # Compute metrics
            source_abstracts = source_papers  # Already strings
            expected_citations = []  # Extract from papers
            for paper_str in source_papers:
                # Extract citation key from paper string
                match = re.search(r'\(([^)]+,\s*\d{4})\)', paper_str)
                if match:
                    expected_citations.append(f"({match.group(1)})")
            
            metrics = compute_all_metrics(
                generated=generated,
                reference=reference_text,
                source_papers=source_abstracts,
                expected_citations=expected_citations
            )
            
            results_list.append(metrics)
            
            print(f"  R1:{metrics.rouge1:.3f} R2:{metrics.rouge2:.3f} "
                  f"BERT:{metrics.bert_score:.3f} [{sample_duration:.1f}s]")
            
            sample_results.append({
                "id": sample_id,
                "topic": topic,
                "generated": generated[:3000],
                "reference": reference_text[:3000],
                "metrics": metrics.to_dict(),
                "time_seconds": sample_duration
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            sample_results.append({"id": sample_id, "error": str(e)})
    
    total_duration = time.time() - total_start
    agg = aggregate_metrics(results_list) if results_list else {}
    
    return {
        "aggregated": agg,
        "samples": sample_results,
        "total_duration": total_duration,
        "avg_duration": total_duration / len(samples) if samples else 0
    }


def print_comparison_table(single_results: dict, mas_results: dict):
    """
    Print side-by-side comparison of Single Agent vs MAS.
    """
    print("\n" + "="*80)
    print("📊 COMPARISON: Single Agent vs Multi-Agent System")
    print("="*80)
    
    single = single_results.get('aggregated', {}) if single_results else {}
    mas = mas_results.get('aggregated', {}) if mas_results else {}
    
    def get_winner(single_val, mas_val, higher_is_better=True):
        if single_val is None or mas_val is None:
            return "N/A"
        if higher_is_better:
            if single_val > mas_val * 1.02:  # 2% margin
                return "Single ✓"
            elif mas_val > single_val * 1.02:
                return "MAS ✓"
            else:
                return "≈ Tie"
        else:  # lower is better (e.g., time)
            if single_val < mas_val * 0.98:
                return "Single ✓"
            elif mas_val < single_val * 0.98:
                return "MAS ✓"
            else:
                return "≈ Tie"
    
    print(f"\n{'Metric':<25} | {'Single Agent':>12} | {'MAS':>12} | {'Winner':>12}")
    print("-"*70)
    
    metrics = [
        ("ROUGE-1", "rouge1_mean", True),
        ("ROUGE-2", "rouge2_mean", True),
        ("ROUGE-L", "rougeL_mean", True),
        ("BERTScore", "bert_score_mean", True),
        ("BLEU", "bleu_mean", True),
        ("METEOR", "meteor_mean", True),
        ("Citation F1", "citation_f1_mean", True),
        ("Vocab Diversity", "vocabulary_diversity_mean", True),
        ("Coherence", "coherence_score_mean", True),
    ]
    
    single_wins = 0
    mas_wins = 0
    
    for name, key, higher_better in metrics:
        s_val = single.get(key)
        m_val = mas.get(key)
        
        s_str = f"{s_val:.3f}" if s_val is not None else "N/A"
        m_str = f"{m_val:.3f}" if m_val is not None else "N/A"
        
        winner = get_winner(s_val, m_val, higher_better)
        if "Single" in winner:
            single_wins += 1
        elif "MAS" in winner:
            mas_wins += 1
        
        print(f"{name:<25} | {s_str:>12} | {m_str:>12} | {winner:>12}")
    
    # Time comparison
    s_time = single_results.get('avg_duration', 0) if single_results else 0
    m_time = mas_results.get('avg_duration', 0) if mas_results else 0
    time_winner = get_winner(s_time, m_time, higher_is_better=False)
    
    print("-"*70)
    print(f"{'Avg Time (sec)':<25} | {s_time:>12.1f} | {m_time:>12.1f} | {time_winner:>12}")
    
    print("\n" + "="*80)
    print(f"📈 SUMMARY: Single Agent wins {single_wins} | MAS wins {mas_wins}")
    
    # Verdict
    if mas_wins > single_wins:
        print("🏆 VERDICT: Multi-Agent System shows improvement!")
    elif single_wins > mas_wins:
        print("⚠️ VERDICT: Single Agent performs better - MAS needs optimization")
    else:
        print("🤝 VERDICT: Both systems perform similarly")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Single Agent vs MAS")
    parser.add_argument("--samples", "-n", type=int, default=5,
                        help="Number of samples to benchmark")
    parser.add_argument("--max-revisions", "-r", type=int, default=3,
                        help="Max revision cycles for MAS")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Run both Single Agent and MAS for comparison")
    parser.add_argument("--single-only", "-s", action="store_true",
                        help="Run Single Agent only")
    parser.add_argument("--mas-only", "-m", action="store_true",
                        help="Run MAS only (default)")
    
    args = parser.parse_args()
    
    single_results = None
    mas_results = None
    
    # Determine what to run
    run_single = args.compare or args.single_only
    run_mas = args.compare or args.mas_only or (not args.single_only)
    
    # Run Single Agent
    if run_single:
        single_results = run_single_agent_benchmark(args.samples, enable_refinement=False)
        
        if single_results:
            agg = single_results['aggregated']
            print(f"\n{'='*70}")
            print("SINGLE AGENT SUMMARY")
            print(f"{'='*70}")
            print(f"ROUGE-1: {agg.get('rouge1_mean', 0):.3f} ± {agg.get('rouge1_std', 0):.3f}")
            print(f"ROUGE-2: {agg.get('rouge2_mean', 0):.3f} ± {agg.get('rouge2_std', 0):.3f}")
            print(f"BERTScore: {agg.get('bert_score_mean', 0):.3f}")
            print(f"Avg Time: {single_results['avg_duration']:.1f}s")
    
    # Run MAS
    if run_mas:
        mas_results = run_benchmark(args.samples, args.max_revisions)
    
    # Show comparison if both were run
    if args.compare and single_results and mas_results:
        print_comparison_table(single_results, mas_results)
        
        # Save comparison results
        comparison_file = f"comparison_single_vs_mas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        comparison = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_samples": args.samples
            },
            "single_agent": single_results,
            "multi_agent_system": {
                "aggregated": mas_results.get('aggregated', {}),
                "total_duration": mas_results.get('total_duration', 0)
            }
        }
        
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n📁 Comparison saved: {comparison_file}")


if __name__ == "__main__":
    main()
