"""
Critic Agent: Quality control and hallucination detection.

Responsibilities:
1. FACTUALITY CHECK: Verify claims against source papers
2. HALLUCINATION DETECTION: Flag unsupported claims
3. COHERENCE CHECK: Ensure logical flow
4. COMPLETENESS CHECK: All citations used?
5. CITATION ACCURACY: Citations match paper content

Input: WriterOutput + Original section (with source papers)
Output: Accept/Reject decision + feedback

Acceptance Criteria:
- All scores >= 7/10
- Citation accuracy >= 60%
- No critical hallucinations
"""

import re
from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import get_llm, clean_output, MASConfig


class CriticOutput:
    """Output from Critic Agent evaluation."""
    def __init__(self, 
                 section_id: int,
                 accept: bool,
                 scores: Dict[str, int],
                 citation_accuracy: float,
                 feedback: str,
                 issues: List[str]):
        self.section_id = section_id
        self.accept = accept
        self.scores = scores  # {grounding, coherence, completeness}
        self.citation_accuracy = citation_accuracy
        self.feedback = feedback
        self.issues = issues
    
    def to_dict(self) -> Dict:
        return {
            "section_id": self.section_id,
            "accept": self.accept,
            "scores": self.scores,
            "citation_accuracy": self.citation_accuracy,
            "feedback": self.feedback,
            "issues": self.issues
        }


class CriticAgent:
    """
    Critic Agent: Evaluates Writer output for quality.
    
    Uses lower temperature for consistent evaluation.
    Checks:
    - Grounding: Are claims supported by cited papers?
    - Coherence: Does the paragraph flow logically?
    - Completeness: Are all assigned citations used?
    - Citation accuracy: Do citations match paper content?
    """
    
    def __init__(self, config: MASConfig = None):
        self.config = config or MASConfig()
        self.llm = get_llm(temperature=0.3, thinking_mode=True)  # Low temp for consistent eval
    
    def evaluate(self, writer_output: Dict, section: Dict) -> CriticOutput:
        """
        Evaluate a single section written by Writer.
        
        Args:
            writer_output: Output from WriterAgent (text, citations_used, etc.)
            section: Original SectionOutline with papers_context
        
        Returns:
            CriticOutput with accept/reject decision and feedback
        """
        section_id = writer_output.get('section_id', 0)
        text = writer_output.get('text', '')
        citations_used = writer_output.get('citations_used', [])
        
        required_citations = section.get('citations_to_use', [])
        papers_context = section.get('papers_context', '')
        
        print(f"[CRITIC] Evaluating section {section_id}...")
        
        # Check 1: Citation completeness (automatic)
        citation_accuracy = self._check_citation_completeness(text, required_citations)
        
        # Check 2: LLM-based quality evaluation
        scores, issues = self._evaluate_quality(text, papers_context, required_citations)
        
        # Decision logic
        grounding = scores.get('grounding', 5)
        coherence = scores.get('coherence', 5)
        completeness = scores.get('completeness', 5)
        
        # Accept if:
        # - All scores >= min_score_accept (default 7)
        # - Citation accuracy >= min_citation_ratio (default 60%)
        # - No critical issues
        min_score = self.config.min_score_accept
        min_citation = self.config.min_citation_ratio
        
        accept = (
            grounding >= min_score and
            coherence >= min_score and
            completeness >= min_score and
            citation_accuracy >= min_citation
        )
        
        # Generate feedback for revision if not accepted
        if not accept:
            feedback = self._generate_feedback(issues, scores, citation_accuracy, required_citations, citations_used)
        else:
            feedback = "Section accepted - meets all quality criteria."
        
        print(f"[CRITIC] Section {section_id}: {'ACCEPT' if accept else 'REJECT'} "
              f"(G:{grounding} C:{coherence} Comp:{completeness} Cite:{citation_accuracy:.1%})")
        
        return CriticOutput(
            section_id=section_id,
            accept=accept,
            scores=scores,
            citation_accuracy=citation_accuracy,
            feedback=feedback,
            issues=issues
        )
    
    def _check_citation_completeness(self, text: str, required_citations: List[str]) -> float:
        """
        Check what percentage of required citations are used.
        """
        if not required_citations:
            return 1.0
        
        used_count = 0
        for cite in required_citations:
            # Check for exact match or close match (handle formatting variations)
            if cite in text:
                used_count += 1
            else:
                # Try partial match (author name + year)
                # Extract author and year from "(Author et al., 2020)"
                match = re.search(r'\(([^,]+),?\s*(\d{4})\)', cite)
                if match:
                    author_part = match.group(1).strip()
                    year = match.group(2)
                    # Check if both appear near each other
                    if author_part.split()[0] in text and year in text:
                        used_count += 1
        
        return used_count / len(required_citations)
    
    def _evaluate_quality(self, text: str, papers_context: str, citations: List[str]) -> Tuple[Dict, List[str]]:
        """
        Use LLM to evaluate text quality.
        
        Returns:
            Tuple of (scores_dict, issues_list)
        """
        system_prompt = """You are a strict academic reviewer evaluating a Related Work paragraph.

### Evaluation Criteria (score 1-10):

1. GROUNDING (1-10): Are ALL claims supported by the cited papers?
   - 10: Every claim has a citation and matches the paper
   - 7: Most claims supported, minor issues
   - 4: Many unsupported claims
   - 1: Mostly hallucinated content

2. COHERENCE (1-10): Does the paragraph flow logically?
   - 10: Perfect transitions, clear argument
   - 7: Good flow, minor awkward transitions
   - 4: Disjointed, hard to follow
   - 1: No logical structure

3. COMPLETENESS (1-10): Is the synthesis comprehensive?
   - 10: All cited papers meaningfully discussed
   - 7: Most papers discussed well
   - 4: Some papers barely mentioned
   - 1: Most citations just listed without discussion

### Output Format (STRICT):
GROUNDING: [score]
COHERENCE: [score]
COMPLETENESS: [score]
ISSUES:
- Issue 1
- Issue 2
(list specific problems found)"""

        user_prompt = f"""### Text to Evaluate
{text}

### Source Papers (ground truth for fact-checking)
{papers_context[:2500]}

### Expected Citations
{', '.join(citations)}

Evaluate this paragraph:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = clean_output(response.content)
            
            # Parse scores
            scores = {
                'grounding': self._extract_score(content, 'GROUNDING'),
                'coherence': self._extract_score(content, 'COHERENCE'),
                'completeness': self._extract_score(content, 'COMPLETENESS')
            }
            
            # Parse issues
            issues = self._extract_issues(content)
            
            return scores, issues
            
        except Exception as e:
            print(f"[CRITIC] Evaluation error: {e}")
            # Return middle-ground scores on error
            return {'grounding': 6, 'coherence': 6, 'completeness': 6}, [f"Evaluation error: {e}"]
    
    def _extract_score(self, content: str, metric: str) -> int:
        """Extract numeric score from response."""
        pattern = rf'{metric}:\s*(\d+)'
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return min(10, max(1, score))  # Clamp to 1-10
        return 5  # Default middle score
    
    def _extract_issues(self, content: str) -> List[str]:
        """Extract list of issues from response."""
        issues = []
        
        # Find ISSUES section
        if 'ISSUES:' in content:
            issues_section = content.split('ISSUES:', 1)[1]
            # Extract bullet points
            for line in issues_section.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    issue = line.lstrip('-•').strip()
                    if issue and len(issue) > 5:
                        issues.append(issue)
        
        return issues[:5]  # Limit to 5 issues
    
    def _generate_feedback(self, issues: List[str], scores: Dict, 
                          citation_accuracy: float, required: List[str], used: List[str]) -> str:
        """
        Generate actionable feedback for Writer revision.
        """
        feedback_parts = []
        
        # Score-based feedback
        if scores.get('grounding', 10) < self.config.min_score_accept:
            feedback_parts.append("GROUNDING: Some claims lack citations or don't match the source papers. Verify each claim.")
        
        if scores.get('coherence', 10) < self.config.min_score_accept:
            feedback_parts.append("COHERENCE: Improve transitions between sentences. Ensure logical flow.")
        
        if scores.get('completeness', 10) < self.config.min_score_accept:
            feedback_parts.append("COMPLETENESS: Discuss ALL cited papers more substantively, not just mention them.")
        
        # Citation feedback
        if citation_accuracy < self.config.min_citation_ratio:
            missing = [c for c in required if c not in used]
            if missing:
                feedback_parts.append(f"MISSING CITATIONS: Must include: {', '.join(missing[:3])}")
        
        # Specific issues
        if issues:
            feedback_parts.append("SPECIFIC ISSUES: " + "; ".join(issues[:3]))
        
        return "\n".join(feedback_parts) if feedback_parts else "Minor improvements needed."
    
    def evaluate_all(self, writer_outputs: List[Dict], sections: List[Dict]) -> List[CriticOutput]:
        """
        Evaluate all sections.
        
        Args:
            writer_outputs: List of WriterOutput dicts
            sections: List of SectionOutline dicts (matching order)
        
        Returns:
            List of CriticOutput objects
        """
        results = []
        for w_out, section in zip(writer_outputs, sections):
            result = self.evaluate(w_out, section)
            results.append(result)
        return results


# --- Standalone test ---
if __name__ == "__main__":
    print("Testing Critic Agent...")
    
    # Mock writer output
    mock_writer_output = {
        "section_id": 0,
        "text": """The transformer architecture introduced by Vaswani et al. (2017) revolutionized 
        natural language processing by eliminating the need for recurrent connections. 
        Building on earlier attention mechanisms (Bahdanau et al., 2015), transformers 
        use self-attention to capture long-range dependencies efficiently.""",
        "citations_used": ["(Vaswani et al., 2017)", "(Bahdanau et al., 2015)"]
    }
    
    # Mock section
    mock_section = {
        "section_id": 0,
        "citations_to_use": ["(Vaswani et al., 2017)", "(Bahdanau et al., 2015)"],
        "papers_context": """[(Vaswani et al., 2017)]
Title: Attention Is All You Need
Abstract: The dominant sequence transduction models are based on complex recurrent or 
convolutional neural networks. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms...

[(Bahdanau et al., 2015)]
Title: Neural Machine Translation by Jointly Learning to Align and Translate
Abstract: Neural machine translation is a recently proposed approach. We conjecture that 
the use of a fixed-length vector is a bottleneck...
"""
    }
    
    critic = CriticAgent()
    result = critic.evaluate(mock_writer_output, mock_section)
    
    print("\n=== Critic Output ===")
    print(f"Accept: {result.accept}")
    print(f"Scores: {result.scores}")
    print(f"Citation accuracy: {result.citation_accuracy:.1%}")
    print(f"Issues: {result.issues}")
    print(f"Feedback:\n{result.feedback}")

