"""
Planner Agent: First agent in the pipeline.

Responsibilities:
1. RETRIEVAL: Fetch relevant papers (Arxiv API or use provided papers)
2. THEME EXTRACTION: Identify 2-4 major themes from the papers
3. PAPER ASSIGNMENT: Assign each paper to a theme
4. WORKFLOW ROUTING: Prepare structured output for Outliner

Output:
{
    "topic": str,
    "target_abstract": str,
    "themes": [
        {
            "theme_id": 0,
            "theme_name": "Theme description",
            "papers": [paper_dict, ...]
        },
        ...
    ],
    "total_papers": int
}
"""

import json
import arxiv
from typing import List, Dict, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import get_llm, clean_output, format_paper_for_citation, MASConfig


class ThemeOutput(TypedDict):
    theme_id: int
    theme_name: str
    papers: List[Dict]


class PlannerOutput(TypedDict):
    topic: str
    target_abstract: str
    fields_of_study: str
    themes: List[ThemeOutput]
    total_papers: int


class PlannerAgent:
    """
    Planner Agent: Handles retrieval and theme extraction.
    
    Two modes:
    - "benchmark": Uses papers provided in input (for evaluation)
    - "live": Fetches papers from Arxiv API (for real use)
    """
    
    def __init__(self, config: MASConfig = None):
        self.config = config or MASConfig()
        self.llm = get_llm(temperature=0.4, thinking_mode=True)  # Lower temp for structured output
    
    def run(self, 
            topic: str, 
            target_abstract: str = "",
            fields_of_study: str = "",
            provided_papers: List[Dict] = None) -> PlannerOutput:
        """
        Main entry point for Planner Agent.
        
        Args:
            topic: Paper title or research topic
            target_abstract: Abstract of the target paper (optional)
            fields_of_study: Research fields (optional)
            provided_papers: Pre-formatted papers for benchmark mode
                Each paper should have: title, authors, year, abstract, citation_key
        
        Returns:
            PlannerOutput with themes and assigned papers
        """
        print(f"[PLANNER] Processing: {topic[:50]}...")
        
        # Step 1: Retrieve papers
        if self.config.retrieval_mode == "benchmark" and provided_papers:
            papers = provided_papers
            print(f"[PLANNER] Using {len(papers)} provided papers (benchmark mode)")
        else:
            papers = self._retrieve_from_arxiv(topic)
            print(f"[PLANNER] Retrieved {len(papers)} papers from Arxiv")
        
        if not papers:
            return {
                "topic": topic,
                "target_abstract": target_abstract,
                "fields_of_study": fields_of_study,
                "themes": [],
                "total_papers": 0
            }
        
        # Step 2: Extract themes and assign papers
        themes = self._extract_themes(topic, target_abstract, papers)
        
        print(f"[PLANNER] Extracted {len(themes)} themes")
        for theme in themes:
            print(f"  - {theme['theme_name']}: {len(theme['papers'])} papers")
        
        return {
            "topic": topic,
            "target_abstract": target_abstract,
            "fields_of_study": fields_of_study,
            "themes": themes,
            "total_papers": len(papers)
        }
    
    def _retrieve_from_arxiv(self, topic: str, max_results: int = 8) -> List[Dict]:
        """Retrieve papers from Arxiv API."""
        print(f"[PLANNER] Searching Arxiv for: {topic[:50]}...")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        try:
            for result in client.results(search):
                authors = [a.name for a in result.authors]
                year = result.published.year
                
                paper = format_paper_for_citation(
                    title=result.title,
                    authors=authors,
                    year=year,
                    abstract=result.summary
                )
                papers.append(paper)
        except Exception as e:
            print(f"[PLANNER] Arxiv error: {e}")
        
        return papers
    
    def _extract_themes(self, topic: str, target_abstract: str, papers: List[Dict]) -> List[ThemeOutput]:
        """
        Use LLM to extract themes and assign papers.
        
        Returns themes with papers assigned to each.
        """
        # Build papers context
        papers_context = ""
        for i, p in enumerate(papers):
            papers_context += f"\n[Paper {i}] {p.get('citation_key', f'Paper_{i}')}\n"
            papers_context += f"Title: {p.get('title', 'Unknown')}\n"
            papers_context += f"Abstract: {p.get('abstract', '')[:500]}...\n"
        
        # DYNAMIC THEME LIMIT: Prevent over-splitting small paper sets
        # - 1-3 papers: 1 theme (synthesize together)
        # - 4-6 papers: max 2 themes
        # - 7+ papers: max 3 themes
        if len(papers) <= 3:
            max_themes = 1
            theme_guidance = "Put ALL papers in ONE cohesive theme (they form a single coherent story)."
        elif len(papers) <= 6:
            max_themes = 2
            theme_guidance = "Create 1-2 themes. Only split if papers address CLEARLY different sub-problems."
        else:
            max_themes = 3
            theme_guidance = "Create 2-3 themes. Group related papers together."
        
        system_prompt = f"""You are a research analyst specializing in literature review organization.

Your task: Analyze the provided papers and group them into thematic clusters.

### CRITICAL: Theme Limit = {max_themes}
{theme_guidance}

### Rules:
1. Each theme should represent a distinct research direction
2. Every paper MUST be assigned to exactly one theme
3. Theme names should be descriptive (e.g., "Deep Learning for Image Classification")
4. PREFER FEWER, BROADER THEMES over many narrow ones
5. If papers are related, PUT THEM TOGETHER in one theme

### Output Format (STRICT JSON):
{{
    "themes": [
        {{
            "theme_name": "Descriptive theme name",
            "paper_indices": [0, 2, 5],
            "rationale": "Why these papers belong together"
        }}
    ]
}}

Output ONLY valid JSON. No markdown, no explanation outside JSON."""

        user_prompt = f"""### Target Paper
Title: {topic}
{f'Abstract: {target_abstract[:500]}' if target_abstract else ''}

### Papers to Organize ({len(papers)} total)
{papers_context}

### Task
Group these papers into {max_themes} theme(s) maximum.
{theme_guidance}
Ensure every paper index (0 to {len(papers)-1}) appears in exactly one theme.

Output JSON:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = clean_output(response.content)
            
            # Parse JSON
            # Try to extract JSON from response
            json_match = content
            if "```json" in content:
                json_match = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_match = content.split("```")[1].split("```")[0]
            
            # Clean up common issues
            json_match = json_match.strip()
            
            result = json.loads(json_match)
            
            # Build output with actual paper data
            themes = []
            for i, theme_data in enumerate(result.get("themes", [])):
                theme_papers = []
                for idx in theme_data.get("paper_indices", []):
                    if 0 <= idx < len(papers):
                        theme_papers.append(papers[idx])
                
                if theme_papers:  # Only add themes with papers
                    themes.append({
                        "theme_id": i,
                        "theme_name": theme_data.get("theme_name", f"Theme {i+1}"),
                        "papers": theme_papers
                    })
            
            # Ensure we have at least one theme
            if not themes:
                themes = self._fallback_themes(papers)
            
            return themes
            
        except json.JSONDecodeError as e:
            print(f"[PLANNER] JSON parse error: {e}")
            return self._fallback_themes(papers)
        except Exception as e:
            print(f"[PLANNER] Theme extraction error: {e}")
            return self._fallback_themes(papers)
    
    def _fallback_themes(self, papers: List[Dict]) -> List[ThemeOutput]:
        """
        Fallback: If LLM fails, create simple themes based on paper count.
        Split papers into 2-3 groups.
        """
        print("[PLANNER] Using fallback theme assignment")
        
        if len(papers) <= 3:
            # Single theme
            return [{
                "theme_id": 0,
                "theme_name": "Related Approaches",
                "papers": papers
            }]
        else:
            # Split into two themes
            mid = len(papers) // 2
            return [
                {
                    "theme_id": 0,
                    "theme_name": "Foundational Works",
                    "papers": papers[:mid]
                },
                {
                    "theme_id": 1,
                    "theme_name": "Recent Advances",
                    "papers": papers[mid:]
                }
            ]


# --- Standalone test ---
if __name__ == "__main__":
    print("Testing Planner Agent...")
    
    planner = PlannerAgent()
    planner.config.retrieval_mode = "live"  # Use Arxiv
    
    result = planner.run(
        topic="Transformer models for natural language processing",
        target_abstract="We propose a new transformer architecture..."
    )
    
    print("\n=== Planner Output ===")
    print(f"Topic: {result['topic']}")
    print(f"Total papers: {result['total_papers']}")
    print(f"Themes: {len(result['themes'])}")
    for theme in result['themes']:
        print(f"  [{theme['theme_id']}] {theme['theme_name']}: {len(theme['papers'])} papers")

