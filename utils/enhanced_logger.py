"""
Enhanced Logging System for Multi-Agent System.

Captures EVERYTHING from each agent for detailed debugging and analysis:
- Full prompts (system + user)
- Raw LLM outputs (including <think> blocks)
- Cleaned outputs
- Reasoning traces
- Intermediate states
- Timing per operation
- ROUGE scores at each revision step

This enables:
1. Understanding why ROUGE scores are what they are
2. Debugging agent behavior
3. Identifying improvement opportunities
4. Research paper documentation
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from rouge_score import rouge_scorer

# Initialize ROUGE scorer for tracking
_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def quick_rouge(generated: str, reference: str) -> Dict[str, float]:
    """Quick ROUGE calculation for logging."""
    if not generated or not reference:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    scores = _rouge_scorer.score(reference, generated)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


@dataclass
class LLMCallLog:
    """Complete log of a single LLM API call."""
    call_id: str
    timestamp: str
    agent_name: str
    operation: str  # e.g., "theme_extraction", "section_writing", "critique"
    
    # Input
    system_prompt: str
    user_prompt: str
    temperature: float
    
    # Output
    raw_output: str  # Including <think> blocks
    cleaned_output: str  # After cleaning
    thinking_content: str  # Extracted <think> block if any
    
    # Timing
    duration_seconds: float
    
    # Quality tracking (if reference available)
    rouge_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AgentStepLog:
    """Log of an agent's complete step (may involve multiple LLM calls)."""
    step_id: str
    agent_name: str
    timestamp: str
    
    # High-level info
    input_summary: str
    output_summary: str
    
    # Detailed LLM calls
    llm_calls: List[LLMCallLog] = field(default_factory=list)
    
    # Structured output
    structured_output: Dict = field(default_factory=dict)
    
    # Quality at this step
    quality_metrics: Dict = field(default_factory=dict)
    
    # Timing
    total_duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'llm_calls': [call.to_dict() for call in self.llm_calls]
        }


@dataclass
class RevisionCycleLog:
    """Log of a complete revision cycle (Writer -> Critic -> Decision)."""
    cycle_number: int
    section_id: int
    
    # Writer output
    writer_text: str
    writer_citations_used: List[str]
    writer_duration: float
    
    # Critic evaluation
    critic_scores: Dict[str, int]
    critic_issues: List[str]
    critic_feedback: str
    critic_accepted: bool
    critic_duration: float
    
    # Quality tracking
    rouge_before: Dict[str, float] = field(default_factory=dict)
    rouge_after: Dict[str, float] = field(default_factory=dict)
    rouge_improvement: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SampleDetailedLog:
    """Complete detailed log for processing one sample."""
    sample_id: str
    topic: str
    target_abstract: str
    fields_of_study: str
    reference_text: str  # Ground truth for ROUGE tracking
    timestamp: str
    
    # Source papers (for analysis)
    source_papers_summary: List[Dict] = field(default_factory=list)
    
    # Agent logs (in order of execution)
    planner_log: Optional[AgentStepLog] = None
    outliner_log: Optional[AgentStepLog] = None
    writer_logs: List[AgentStepLog] = field(default_factory=list)
    critic_logs: List[AgentStepLog] = field(default_factory=list)
    assembly_log: Optional[AgentStepLog] = None
    
    # Revision tracking (crucial for understanding ROUGE progression)
    revision_cycles: List[RevisionCycleLog] = field(default_factory=list)
    
    # Final outputs
    final_text: str = ""
    final_rouge: Dict[str, float] = field(default_factory=dict)
    
    # All metrics
    final_metrics: Dict = field(default_factory=dict)
    
    # Total timing
    total_duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'sample_id': self.sample_id,
            'topic': self.topic,
            'target_abstract': self.target_abstract[:500] + '...' if len(self.target_abstract) > 500 else self.target_abstract,
            'fields_of_study': self.fields_of_study,
            'reference_text': self.reference_text,
            'timestamp': self.timestamp,
            'source_papers_summary': self.source_papers_summary,
            'planner_log': self.planner_log.to_dict() if self.planner_log else None,
            'outliner_log': self.outliner_log.to_dict() if self.outliner_log else None,
            'writer_logs': [w.to_dict() for w in self.writer_logs],
            'critic_logs': [c.to_dict() for c in self.critic_logs],
            'assembly_log': self.assembly_log.to_dict() if self.assembly_log else None,
            'revision_cycles': [r.to_dict() for r in self.revision_cycles],
            'final_text': self.final_text,
            'final_rouge': self.final_rouge,
            'final_metrics': self.final_metrics,
            'total_duration_seconds': self.total_duration_seconds
        }


class EnhancedMASLogger:
    """
    Enhanced logger that captures EVERYTHING for debugging and improvement.
    
    Key features:
    - Tracks ROUGE at every revision step
    - Captures full prompts and raw outputs
    - Extracts and preserves <think> reasoning blocks
    - Tracks timing for performance analysis
    - Provides summary statistics
    """
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Storage
        self.samples: Dict[str, SampleDetailedLog] = {}
        self.current_sample_id: Optional[str] = None
        self._call_counter = 0
        
        # Experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'summary_stats': {}
        }
        
        print(f"[ENHANCED_LOGGER] Initialized: {experiment_name}")
        print(f"[ENHANCED_LOGGER] Log file will be: {log_dir}/{experiment_name}_{self.timestamp}_detailed.json")
    
    def set_config(self, config: Dict):
        """Store experiment configuration."""
        self.metadata['config'] = config
    
    def start_sample(self, sample_id: str, topic: str, target_abstract: str = "",
                     fields_of_study: str = "", reference_text: str = "",
                     source_papers: List[Dict] = None):
        """Start logging for a new sample."""
        self.current_sample_id = sample_id
        
        # Summarize source papers
        papers_summary = []
        if source_papers:
            for p in source_papers:
                papers_summary.append({
                    'citation_key': p.get('citation_key', 'Unknown'),
                    'title': p.get('title', 'Unknown')[:100],
                    'year': p.get('year', 'Unknown')
                })
        
        self.samples[sample_id] = SampleDetailedLog(
            sample_id=sample_id,
            topic=topic,
            target_abstract=target_abstract,
            fields_of_study=fields_of_study,
            reference_text=reference_text,
            timestamp=datetime.now().isoformat(),
            source_papers_summary=papers_summary
        )
        
        print(f"\n[ENHANCED_LOGGER] === Started Sample: {sample_id} ===")
        print(f"[ENHANCED_LOGGER] Topic: {topic[:60]}...")
        print(f"[ENHANCED_LOGGER] Reference length: {len(reference_text)} chars")
    
    def log_llm_call(self, agent_name: str, operation: str,
                     system_prompt: str, user_prompt: str,
                     raw_output: str, cleaned_output: str,
                     temperature: float, duration: float,
                     reference_text: str = None) -> LLMCallLog:
        """Log a single LLM API call with full details."""
        self._call_counter += 1
        call_id = f"{agent_name}_{operation}_{self._call_counter}"
        
        # Extract thinking content if present
        import re
        thinking_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
        thinking_content = thinking_match.group(1).strip() if thinking_match else ""
        
        # Calculate ROUGE if reference available
        rouge_scores = {}
        if reference_text and cleaned_output:
            rouge_scores = quick_rouge(cleaned_output, reference_text)
        
        call_log = LLMCallLog(
            call_id=call_id,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            operation=operation,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            raw_output=raw_output,
            cleaned_output=cleaned_output,
            thinking_content=thinking_content[:2000] if thinking_content else "",  # Truncate long thinking
            duration_seconds=duration,
            rouge_scores=rouge_scores
        )
        
        # Print summary
        rouge_str = f" R1:{rouge_scores.get('rouge1', 0):.3f}" if rouge_scores else ""
        print(f"[ENHANCED_LOGGER] LLM Call: {agent_name}.{operation} ({duration:.2f}s){rouge_str}")
        if thinking_content:
            print(f"[ENHANCED_LOGGER]   └─ Thinking: {len(thinking_content)} chars captured")
        
        return call_log
    
    def log_planner(self, themes: List[Dict], papers_count: int,
                    llm_calls: List[LLMCallLog] = None, duration: float = 0.0):
        """Log Planner Agent execution."""
        if not self.current_sample_id:
            return
        
        sample = self.samples[self.current_sample_id]
        
        step_log = AgentStepLog(
            step_id=f"planner_{self.current_sample_id}",
            agent_name="Planner",
            timestamp=datetime.now().isoformat(),
            input_summary=f"Topic: {sample.topic[:50]}...",
            output_summary=f"Extracted {len(themes)} themes from {papers_count} papers",
            llm_calls=llm_calls or [],
            structured_output={
                'themes': [{'name': t.get('theme_name', ''), 'papers_count': len(t.get('papers', []))} for t in themes],
                'total_papers': papers_count
            },
            total_duration_seconds=duration
        )
        
        sample.planner_log = step_log
        
        print(f"[ENHANCED_LOGGER] Planner: {len(themes)} themes, {papers_count} papers ({duration:.2f}s)")
        for t in themes:
            print(f"[ENHANCED_LOGGER]   └─ {t.get('theme_name', 'Unknown')[:40]}: {len(t.get('papers', []))} papers")
    
    def log_outliner(self, sections: List[Dict], gap_hint: str,
                     llm_calls: List[LLMCallLog] = None, duration: float = 0.0):
        """Log Outliner Agent execution."""
        if not self.current_sample_id:
            return
        
        sample = self.samples[self.current_sample_id]
        
        step_log = AgentStepLog(
            step_id=f"outliner_{self.current_sample_id}",
            agent_name="Outliner",
            timestamp=datetime.now().isoformat(),
            input_summary=f"{len(sections)} themes to outline",
            output_summary=f"Created {len(sections)} sections",
            llm_calls=llm_calls or [],
            structured_output={
                'sections': [{'theme': s.get('theme_name', ''), 'citations': len(s.get('citations_to_use', []))} for s in sections],
                'gap_hint': gap_hint[:200]
            },
            total_duration_seconds=duration
        )
        
        sample.outliner_log = step_log
        
        print(f"[ENHANCED_LOGGER] Outliner: {len(sections)} sections ({duration:.2f}s)")
    
    def log_writer(self, section_id: int, theme_name: str,
                   generated_text: str, citations_used: List[str],
                   revision_number: int = 0,
                   llm_calls: List[LLMCallLog] = None,
                   duration: float = 0.0):
        """Log Writer Agent execution with ROUGE tracking."""
        if not self.current_sample_id:
            return
        
        sample = self.samples[self.current_sample_id]
        
        # Calculate current ROUGE
        rouge = quick_rouge(generated_text, sample.reference_text)
        
        step_log = AgentStepLog(
            step_id=f"writer_s{section_id}_r{revision_number}_{self.current_sample_id}",
            agent_name="Writer",
            timestamp=datetime.now().isoformat(),
            input_summary=f"Section {section_id}: {theme_name[:40]}...",
            output_summary=f"{len(generated_text)} chars, {len(citations_used)} citations",
            llm_calls=llm_calls or [],
            structured_output={
                'section_id': section_id,
                'theme_name': theme_name,
                'generated_text': generated_text,
                'citations_used': citations_used,
                'revision_number': revision_number
            },
            quality_metrics=rouge,
            total_duration_seconds=duration
        )
        
        sample.writer_logs.append(step_log)
        
        cite_pct = len(citations_used) * 100 // max(1, len(citations_used) + 2)  # Rough estimate
        print(f"[ENHANCED_LOGGER] Writer S{section_id} R{revision_number}: "
              f"{len(generated_text)} chars | R1:{rouge['rouge1']:.3f} R2:{rouge['rouge2']:.3f} RL:{rouge['rougeL']:.3f}")
    
    def log_critic(self, section_id: int, scores: Dict[str, int],
                   issues: List[str], accepted: bool, feedback: str,
                   citation_accuracy: float,
                   llm_calls: List[LLMCallLog] = None,
                   duration: float = 0.0):
        """Log Critic Agent evaluation."""
        if not self.current_sample_id:
            return
        
        sample = self.samples[self.current_sample_id]
        
        step_log = AgentStepLog(
            step_id=f"critic_s{section_id}_{self.current_sample_id}",
            agent_name="Critic",
            timestamp=datetime.now().isoformat(),
            input_summary=f"Evaluating Section {section_id}",
            output_summary=f"{'ACCEPT' if accepted else 'REJECT'} - G:{scores.get('grounding', 0)} C:{scores.get('coherence', 0)}",
            llm_calls=llm_calls or [],
            structured_output={
                'section_id': section_id,
                'scores': scores,
                'issues': issues,
                'accepted': accepted,
                'feedback': feedback,
                'citation_accuracy': citation_accuracy
            },
            total_duration_seconds=duration
        )
        
        sample.critic_logs.append(step_log)
        
        status = "✓ ACCEPT" if accepted else "✗ REJECT"
        print(f"[ENHANCED_LOGGER] Critic S{section_id}: {status} | "
              f"G:{scores.get('grounding', 0)} C:{scores.get('coherence', 0)} Cite:{citation_accuracy:.0%}")
        if issues:
            for issue in issues[:2]:
                print(f"[ENHANCED_LOGGER]   └─ Issue: {issue[:60]}...")
    
    def log_revision_cycle(self, section_id: int, cycle_number: int,
                           writer_text: str, writer_citations: List[str],
                           writer_duration: float,
                           critic_scores: Dict, critic_issues: List[str],
                           critic_feedback: str, critic_accepted: bool,
                           critic_duration: float,
                           previous_text: str = ""):
        """Log a complete revision cycle with ROUGE progression."""
        if not self.current_sample_id:
            return
        
        sample = self.samples[self.current_sample_id]
        
        # Calculate ROUGE before and after
        rouge_before = quick_rouge(previous_text, sample.reference_text) if previous_text else {}
        rouge_after = quick_rouge(writer_text, sample.reference_text)
        
        # Calculate improvement
        rouge_improvement = {}
        if rouge_before:
            for key in ['rouge1', 'rouge2', 'rougeL']:
                rouge_improvement[key] = rouge_after.get(key, 0) - rouge_before.get(key, 0)
        
        cycle_log = RevisionCycleLog(
            cycle_number=cycle_number,
            section_id=section_id,
            writer_text=writer_text,
            writer_citations_used=writer_citations,
            writer_duration=writer_duration,
            critic_scores=critic_scores,
            critic_issues=critic_issues,
            critic_feedback=critic_feedback,
            critic_accepted=critic_accepted,
            critic_duration=critic_duration,
            rouge_before=rouge_before,
            rouge_after=rouge_after,
            rouge_improvement=rouge_improvement
        )
        
        sample.revision_cycles.append(cycle_log)
        
        # Print ROUGE progression
        if rouge_before:
            r1_delta = rouge_improvement.get('rouge1', 0)
            sign = "+" if r1_delta >= 0 else ""
            print(f"[ENHANCED_LOGGER] Revision S{section_id} C{cycle_number}: "
                  f"R1 {rouge_before.get('rouge1', 0):.3f} → {rouge_after.get('rouge1', 0):.3f} ({sign}{r1_delta:.3f})")
    
    def log_assembly(self, sections_combined: int, final_text: str,
                     llm_calls: List[LLMCallLog] = None,
                     duration: float = 0.0):
        """Log Final Assembly execution."""
        if not self.current_sample_id:
            return
        
        sample = self.samples[self.current_sample_id]
        
        # Calculate final ROUGE
        final_rouge = quick_rouge(final_text, sample.reference_text)
        
        step_log = AgentStepLog(
            step_id=f"assembly_{self.current_sample_id}",
            agent_name="Assembly",
            timestamp=datetime.now().isoformat(),
            input_summary=f"Combining {sections_combined} sections",
            output_summary=f"Final output: {len(final_text)} chars",
            llm_calls=llm_calls or [],
            structured_output={
                'sections_combined': sections_combined,
                'final_text_length': len(final_text)
            },
            quality_metrics=final_rouge,
            total_duration_seconds=duration
        )
        
        sample.assembly_log = step_log
        sample.final_text = final_text
        sample.final_rouge = final_rouge
        
        print(f"[ENHANCED_LOGGER] Assembly: {sections_combined} sections → {len(final_text)} chars")
        print(f"[ENHANCED_LOGGER]   └─ Final ROUGE: R1:{final_rouge['rouge1']:.3f} "
              f"R2:{final_rouge['rouge2']:.3f} RL:{final_rouge['rougeL']:.3f}")
    
    def finalize_sample(self, final_metrics: Dict, total_duration: float):
        """Finalize a sample with all metrics."""
        if not self.current_sample_id:
            return
        
        sample = self.samples[self.current_sample_id]
        sample.final_metrics = final_metrics
        sample.total_duration_seconds = total_duration
        
        # Calculate revision stats
        total_revisions = len(sample.revision_cycles)
        accepted_first_try = sum(1 for c in sample.revision_cycles if c.cycle_number == 0 and c.critic_accepted)
        
        print(f"\n[ENHANCED_LOGGER] === Sample Complete: {self.current_sample_id} ===")
        print(f"[ENHANCED_LOGGER] Total Time: {total_duration:.2f}s")
        print(f"[ENHANCED_LOGGER] Revision Cycles: {total_revisions}")
        print(f"[ENHANCED_LOGGER] Final ROUGE-1: {sample.final_rouge.get('rouge1', 0):.3f}")
        
        self.current_sample_id = None
    
    def save(self, filename: str = None) -> str:
        """Save all detailed logs to JSON file."""
        if filename is None:
            filename = f"{self.log_dir}/{self.experiment_name}_{self.timestamp}_detailed.json"
        
        # Calculate summary statistics
        self._calculate_summary_stats()
        
        output = {
            'metadata': self.metadata,
            'samples': {sid: sample.to_dict() for sid, sample in self.samples.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[ENHANCED_LOGGER] Saved detailed logs to: {filename}")
        print(f"[ENHANCED_LOGGER] Total samples: {len(self.samples)}")
        
        return filename
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics across all samples."""
        if not self.samples:
            return
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        total_revisions = 0
        llm_calls_total = 0
        total_duration = 0.0
        
        for sample in self.samples.values():
            if sample.final_rouge:
                rouge1_scores.append(sample.final_rouge.get('rouge1', 0))
                rouge2_scores.append(sample.final_rouge.get('rouge2', 0))
                rougeL_scores.append(sample.final_rouge.get('rougeL', 0))
            
            total_revisions += len(sample.revision_cycles)
            total_duration += sample.total_duration_seconds
            
            # Count LLM calls
            if sample.planner_log:
                llm_calls_total += len(sample.planner_log.llm_calls)
            if sample.outliner_log:
                llm_calls_total += len(sample.outliner_log.llm_calls)
            for w in sample.writer_logs:
                llm_calls_total += len(w.llm_calls)
            for c in sample.critic_logs:
                llm_calls_total += len(c.llm_calls)
            if sample.assembly_log:
                llm_calls_total += len(sample.assembly_log.llm_calls)
        
        import numpy as np
        
        self.metadata['summary_stats'] = {
            'total_samples': len(self.samples),
            'rouge1_mean': float(np.mean(rouge1_scores)) if rouge1_scores else 0,
            'rouge1_std': float(np.std(rouge1_scores)) if rouge1_scores else 0,
            'rouge2_mean': float(np.mean(rouge2_scores)) if rouge2_scores else 0,
            'rouge2_std': float(np.std(rouge2_scores)) if rouge2_scores else 0,
            'rougeL_mean': float(np.mean(rougeL_scores)) if rougeL_scores else 0,
            'rougeL_std': float(np.std(rougeL_scores)) if rougeL_scores else 0,
            'total_revision_cycles': total_revisions,
            'avg_revisions_per_sample': total_revisions / len(self.samples),
            'total_llm_calls': llm_calls_total,
            'avg_llm_calls_per_sample': llm_calls_total / len(self.samples),
            'total_duration_seconds': total_duration,
            'avg_duration_per_sample': total_duration / len(self.samples)
        }


# Singleton for easy access
_global_logger: Optional[EnhancedMASLogger] = None


def get_logger() -> Optional[EnhancedMASLogger]:
    """Get the global logger instance."""
    return _global_logger


def set_logger(logger: EnhancedMASLogger):
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger


# --- Standalone test ---
if __name__ == "__main__":
    print("Testing Enhanced Logger...")
    
    logger = EnhancedMASLogger("test_detailed")
    
    logger.start_sample(
        sample_id="test_1",
        topic="Transformer models for NLP",
        reference_text="This is the reference text for testing ROUGE calculations."
    )
    
    # Log a mock LLM call
    call = logger.log_llm_call(
        agent_name="Writer",
        operation="section_writing",
        system_prompt="You are an expert...",
        user_prompt="Write about transformers...",
        raw_output="<think>Let me analyze this...</think>\n\nTransformers have revolutionized NLP.",
        cleaned_output="Transformers have revolutionized NLP.",
        temperature=0.6,
        duration=2.5,
        reference_text="This is the reference text for testing ROUGE calculations."
    )
    
    logger.log_planner(
        themes=[{"theme_name": "Attention Mechanisms", "papers": [{}]}],
        papers_count=5,
        duration=3.0
    )
    
    logger.finalize_sample(
        final_metrics={"rouge1": 0.35},
        total_duration=15.0
    )
    
    logger.save()
    print("\nTest complete!")

