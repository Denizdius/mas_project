"""
Writer Agent: Generates draft paragraphs.

Responsibilities:
1. PARAGRAPH GENERATION: Write one paragraph per section
2. CITATION USAGE: Use ALL assigned citations
3. SCIENTIFIC TONE: Formal academic language
4. REASONING: Qwen3 thinking mode for complex synthesis

Input: Single SectionOutline from Outliner
Output: Draft paragraph text

Can run in PARALLEL for multiple sections.
"""

from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
import concurrent.futures

from agents.base import get_llm, clean_output, MASConfig


class WriterOutput:
    """Output from Writer Agent for one section."""
    def __init__(self, section_id: int, text: str, citations_used: List[str], revision_count: int = 0):
        self.section_id = section_id
        self.text = text
        self.citations_used = citations_used
        self.revision_count = revision_count
    
    def to_dict(self) -> Dict:
        return {
            "section_id": self.section_id,
            "text": self.text,
            "citations_used": self.citations_used,
            "revision_count": self.revision_count
        }


class WriterAgent:
    """
    Writer Agent: Generates academic paragraphs from outlines.
    
    Key features:
    - Uses Qwen3 thinking mode for complex reasoning
    - Enforces citation usage from outline
    - Supports parallel writing for multiple sections
    - Can revise based on Critic feedback
    """
    
    def __init__(self, config: MASConfig = None):
        self.config = config or MASConfig()
        self.llm = get_llm(temperature=0.6, thinking_mode=True)  # Creative but controlled
    
    def write_section(self, section: Dict, topic: str, is_last: bool = False, 
                      feedback: str = None) -> WriterOutput:
        """
        Write a single section/paragraph.
        
        Args:
            section: SectionOutline from Outliner
            topic: Overall paper topic
            is_last: Whether this is the last section (add gap analysis)
            feedback: Optional feedback from Critic for revision
        
        Returns:
            WriterOutput with paragraph text
        """
        section_id = section.get('section_id', 0)
        theme_name = section.get('theme_name', 'Related Work')
        key_point = section.get('key_point', '')
        citations = section.get('citations_to_use', [])
        papers_context = section.get('papers_context', '')
        transition_hint = section.get('transition_hint', '')
        
        print(f"[WRITER] Writing section {section_id}: {theme_name[:30]}...")
        
        # Build prompt
        citations_str = ", ".join(citations)
        
        # Length constraint based on number of citations
        num_citations = len(citations)
        if num_citations <= 2:
            word_limit = "100-150 words"
            max_sentences = 5
        elif num_citations <= 4:
            word_limit = "150-200 words"
            max_sentences = 7
        else:
            word_limit = "200-250 words"
            max_sentences = 9
        
        system_prompt = f"""You are an expert academic writer specializing in literature reviews.

### Your Task
Write ONE CONCISE paragraph synthesizing the provided papers.

### STRICT LENGTH: {word_limit} maximum, {max_sentences} sentences max
SHORTER IS BETTER. Be concise. Every sentence must add value.

### Critical Rules
1. USE ALL CITATIONS: Every citation in the list MUST appear in your paragraph
2. TRUE SYNTHESIS: Compare/contrast papers, don't just describe each one separately
3. CITATION FORMAT: Use (Author, Year) or (Author et al., Year) - EXACT format provided
4. NO FILLER: No generic phrases like "research shows" or "studies indicate"
5. ACADEMIC TONE: Formal, third-person, objective language
6. ONE PARAGRAPH: No line breaks, no headers

### What TRUE SYNTHESIS looks like:
BAD: "Author1 did X. Author2 did Y. Author3 did Z."
GOOD: "While Author1 and Author2 both address X, they differ in Y (Author1) vs Z (Author2)."

### Output
Write ONLY the paragraph. No preamble, no explanation, no commentary."""

        user_prompt = f"""### Paper Topic
{topic}

### Section Theme
{theme_name}

### Citations to Use (ALL REQUIRED)
{citations_str}

### Papers (synthesize these - compare/contrast their approaches)
{papers_context[:2500]}

### Key Point
{key_point}

### REMEMBER: {word_limit}, TRUE SYNTHESIS (compare papers, don't list them)
"""
        
        # Add revision context if feedback provided
        if feedback:
            user_prompt += f"""
### REVISION NEEDED
Previous version had these issues:
{feedback}

Fix ALL issues while maintaining the key point and using ALL citations.
"""

        # Add gap analysis instruction for last section
        if is_last:
            user_prompt += """
### Final Section
This is the LAST paragraph. End with 1-2 sentences identifying gaps in the existing work
that motivate the current paper.
"""

        user_prompt += "\n\nWrite the paragraph now:"

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            text = clean_output(response.content)
            
            # Verify citations used
            citations_used = [c for c in citations if c in text]
            
            return WriterOutput(
                section_id=section_id,
                text=text,
                citations_used=citations_used,
                revision_count=0 if not feedback else 1
            )
            
        except Exception as e:
            print(f"[WRITER] Error in section {section_id}: {e}")
            return WriterOutput(
                section_id=section_id,
                text=f"[Error generating section: {str(e)}]",
                citations_used=[],
                revision_count=0
            )
    
    def write_all_sections(self, outliner_output: Dict, parallel: bool = True) -> List[WriterOutput]:
        """
        Write all sections from Outliner output.
        
        Args:
            outliner_output: Output from OutlinerAgent.run()
            parallel: Whether to write sections in parallel
        
        Returns:
            List of WriterOutput objects
        """
        topic = outliner_output.get('topic', '')
        sections = outliner_output.get('sections', [])
        
        if not sections:
            return []
        
        print(f"[WRITER] Writing {len(sections)} sections {'(parallel)' if parallel else '(sequential)'}...")
        
        results = []
        
        if parallel and self.config.enable_parallel and len(sections) > 1:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(sections))) as executor:
                futures = {}
                for i, section in enumerate(sections):
                    is_last = (i == len(sections) - 1)
                    future = executor.submit(self.write_section, section, topic, is_last)
                    futures[future] = i
                
                # Collect results in order
                results = [None] * len(sections)
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"[WRITER] Parallel task {idx} failed: {e}")
                        results[idx] = WriterOutput(idx, f"[Error: {e}]", [], 0)
        else:
            # Sequential execution
            for i, section in enumerate(sections):
                is_last = (i == len(sections) - 1)
                result = self.write_section(section, topic, is_last)
                results.append(result)
        
        return results
    
    def revise_section(self, section: Dict, topic: str, previous_output: WriterOutput, 
                       feedback: str, is_last: bool = False) -> WriterOutput:
        """
        Revise a section based on Critic feedback.
        
        Args:
            section: Original SectionOutline
            topic: Paper topic
            previous_output: Previous WriterOutput that needs revision
            feedback: Specific feedback from Critic
            is_last: Whether this is the last section
        
        Returns:
            Revised WriterOutput with incremented revision_count
        """
        print(f"[WRITER] Revising section {section.get('section_id', 0)} (attempt {previous_output.revision_count + 1})...")
        
        # Include previous text in context
        section_with_prev = dict(section)
        section_with_prev['previous_attempt'] = previous_output.text
        
        result = self.write_section(section, topic, is_last, feedback=feedback)
        result.revision_count = previous_output.revision_count + 1
        
        return result


# --- Standalone test ---
if __name__ == "__main__":
    print("Testing Writer Agent...")
    
    # Mock section from Outliner
    mock_section = {
        "section_id": 0,
        "theme_name": "Attention Mechanisms in NLP",
        "key_point": "Attention mechanisms have revolutionized sequence modeling by allowing models to focus on relevant parts of the input.",
        "citations_to_use": ["(Vaswani et al., 2017)", "(Bahdanau et al., 2015)"],
        "papers_context": """[(Vaswani et al., 2017)]
Title: Attention Is All You Need
Authors: Ashish Vaswani, Noam Shazeer, ...
Year: 2017
Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...

[(Bahdanau et al., 2015)]
Title: Neural Machine Translation by Jointly Learning to Align and Translate
Authors: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
Year: 2015
Abstract: Neural machine translation is a recently proposed approach to machine translation...
""",
        "transition_hint": "Connect to efficient transformers"
    }
    
    writer = WriterAgent()
    result = writer.write_section(mock_section, "Efficient Transformers for NLP", is_last=False)
    
    print("\n=== Writer Output ===")
    print(f"Section ID: {result.section_id}")
    print(f"Citations used: {result.citations_used}")
    print(f"Text:\n{result.text}")

