"""
Outliner Agent: Creates detailed section structure.

Responsibilities:
1. SECTION STRUCTURE: Define paragraph structure for each theme
2. CITATION ANCHORS: Pre-assign which citations go where
3. TRANSITION PLANNING: Plan how sections connect
4. DETAILED OUTLINE: Produce actionable outline for Writer

Input: PlannerOutput (themes with papers)
Output: Outline with citation anchors per section
"""

import json
from typing import List, Dict, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import get_llm, clean_output, MASConfig


class SectionOutline(TypedDict):
    section_id: int
    theme_name: str
    key_point: str  # What this section should argue
    citations_to_use: List[str]  # Citation keys to include
    papers_context: str  # Full paper info for Writer
    transition_hint: str  # How to connect to next section


class OutlinerOutput(TypedDict):
    topic: str
    target_abstract: str
    sections: List[SectionOutline]
    gap_analysis_hint: str  # Hint for final gap analysis


class OutlinerAgent:
    """
    Outliner Agent: Creates structured outline with citation anchors.
    
    Key innovation: Pre-assigns citations BEFORE writing.
    This ensures better grounding and prevents citation hallucination.
    """
    
    def __init__(self, config: MASConfig = None):
        self.config = config or MASConfig()
        self.llm = get_llm(temperature=0.4, thinking_mode=True)
    
    def run(self, planner_output: Dict) -> OutlinerOutput:
        """
        Create detailed outline from Planner's themes.
        
        Args:
            planner_output: Output from PlannerAgent.run()
        
        Returns:
            OutlinerOutput with sections and citation anchors
        """
        print(f"[OUTLINER] Creating outline for {len(planner_output.get('themes', []))} themes...")
        
        topic = planner_output.get('topic', '')
        target_abstract = planner_output.get('target_abstract', '')
        themes = planner_output.get('themes', [])
        
        if not themes:
            return {
                "topic": topic,
                "target_abstract": target_abstract,
                "sections": [],
                "gap_analysis_hint": "No papers to analyze"
            }
        
        # Generate outline for each theme
        sections = self._create_sections(topic, target_abstract, themes)
        
        # Generate gap analysis hint
        gap_hint = self._generate_gap_hint(topic, target_abstract, themes)
        
        print(f"[OUTLINER] Created {len(sections)} sections")
        for s in sections:
            print(f"  - Section {s['section_id']}: {s['theme_name'][:40]}... ({len(s['citations_to_use'])} citations)")
        
        return {
            "topic": topic,
            "target_abstract": target_abstract,
            "sections": sections,
            "gap_analysis_hint": gap_hint
        }
    
    def _create_sections(self, topic: str, target_abstract: str, themes: List[Dict]) -> List[SectionOutline]:
        """
        Create detailed section outlines with citation anchors.
        """
        sections = []
        
        for i, theme in enumerate(themes):
            theme_name = theme.get('theme_name', f'Theme {i+1}')
            papers = theme.get('papers', [])
            
            if not papers:
                continue
            
            # Collect citation keys
            citation_keys = [p.get('citation_key', f'(Unknown, 20XX)') for p in papers]
            
            # Build papers context for Writer
            papers_context = ""
            for p in papers:
                papers_context += p.get('full_text', '') + "\n"
            
            # Generate key point using LLM
            key_point = self._generate_key_point(topic, theme_name, papers)
            
            # Transition hint
            if i < len(themes) - 1:
                next_theme = themes[i + 1].get('theme_name', 'next topic')
                transition_hint = f"Connect to {next_theme}"
            else:
                transition_hint = "Lead into gap analysis"
            
            sections.append({
                "section_id": i,
                "theme_name": theme_name,
                "key_point": key_point,
                "citations_to_use": citation_keys,
                "papers_context": papers_context[:self.config.max_section_context],
                "transition_hint": transition_hint
            })
        
        return sections
    
    def _generate_key_point(self, topic: str, theme_name: str, papers: List[Dict]) -> str:
        """
        Generate the key argument/point for a section.
        """
        # Build brief paper summary
        papers_brief = ""
        for p in papers[:5]:  # Limit to 5 papers
            papers_brief += f"- {p.get('title', 'Unknown')}: {p.get('abstract', '')[:200]}...\n"
        
        prompt = f"""Given these papers under the theme "{theme_name}" for a paper about "{topic}":

{papers_brief}

What is the ONE key point this section should make? (1-2 sentences)
Focus on synthesizing the papers' contributions, not listing them.

Key point:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert at synthesizing academic literature. Be concise."),
                HumanMessage(content=prompt)
            ])
            
            key_point = clean_output(response.content)
            # Truncate if too long
            if len(key_point) > 300:
                key_point = key_point[:300] + "..."
            return key_point
            
        except Exception as e:
            print(f"[OUTLINER] Key point generation error: {e}")
            return f"Discuss {theme_name} approaches and their contributions."
    
    def _generate_gap_hint(self, topic: str, target_abstract: str, themes: List[Dict]) -> str:
        """
        Generate a hint for the gap analysis paragraph.
        """
        # Collect all paper titles
        all_titles = []
        for theme in themes:
            for p in theme.get('papers', []):
                all_titles.append(p.get('title', ''))
        
        prompt = f"""For a paper titled "{topic}":
{f'About: {target_abstract[:300]}' if target_abstract else ''}

The related work covers these topics:
{', '.join([t.get('theme_name', '') for t in themes])}

What GAP or limitation in the existing work does this paper address?
(1-2 sentences suggesting what's missing)

Gap:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert at identifying research gaps. Be specific and concise."),
                HumanMessage(content=prompt)
            ])
            
            gap_hint = clean_output(response.content)
            if len(gap_hint) > 300:
                gap_hint = gap_hint[:300] + "..."
            return gap_hint
            
        except Exception as e:
            print(f"[OUTLINER] Gap hint error: {e}")
            return "Despite these advances, challenges remain in combining these approaches effectively."


# --- Standalone test ---
if __name__ == "__main__":
    print("Testing Outliner Agent...")
    
    # Mock planner output
    mock_planner_output = {
        "topic": "Transformer models for NLP",
        "target_abstract": "We propose a new efficient transformer...",
        "themes": [
            {
                "theme_id": 0,
                "theme_name": "Attention Mechanisms",
                "papers": [
                    {
                        "citation_key": "(Vaswani et al., 2017)",
                        "title": "Attention Is All You Need",
                        "abstract": "The dominant sequence transduction models...",
                        "full_text": "[Vaswani et al., 2017]\nTitle: Attention Is All You Need\n..."
                    }
                ]
            },
            {
                "theme_id": 1,
                "theme_name": "Efficient Transformers",
                "papers": [
                    {
                        "citation_key": "(Kitaev et al., 2020)",
                        "title": "Reformer: The Efficient Transformer",
                        "abstract": "Large Transformer models routinely...",
                        "full_text": "[Kitaev et al., 2020]\nTitle: Reformer...\n..."
                    }
                ]
            }
        ]
    }
    
    outliner = OutlinerAgent()
    result = outliner.run(mock_planner_output)
    
    print("\n=== Outliner Output ===")
    print(f"Sections: {len(result['sections'])}")
    for s in result['sections']:
        print(f"\n[Section {s['section_id']}] {s['theme_name']}")
        print(f"  Key point: {s['key_point'][:100]}...")
        print(f"  Citations: {s['citations_to_use']}")
        print(f"  Transition: {s['transition_hint']}")
    print(f"\nGap hint: {result['gap_analysis_hint']}")

