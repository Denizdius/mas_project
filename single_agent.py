"""
Single Agent v2: Improved Related Work Generator

Improvements over v1:
1. Uses (Author, Year) citation format to match OARelatedWork references
2. Self-critique refinement loop for better quality
3. Optimized prompts for higher ROUGE scores
4. Qwen3 thinking mode parameters (temperature=0.6, top_p=0.95)
5. Optimized for 16K context window

NOTE: Start vLLM with thinking mode enabled:
  vllm serve Qwen/Qwen3-8B-quantized.w4a16 --enable-reasoning --reasoning-parser deepseek_r1
"""

import os
import arxiv
import re
from typing import List, TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- Configuration ---
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "/home/deniz/Documents/mas_project/Qwen3-8B-quantized.w4a16"

# Context window optimization (16K tokens available)
MAX_PAPERS_CONTEXT = 12000  # Characters for source papers
MAX_EXAMPLES_CONTEXT = 3000  # Characters for few-shot examples
MAX_REFINEMENT_CONTEXT = 4000  # Characters for self-critique

# --- State Definition ---
class AgentState(TypedDict):
    topic: str
    papers: List[str]
    draft: Optional[str]
    refined_draft: Optional[str]
    examples: Optional[List[dict]]
    enable_refinement: bool  # Whether to run self-critique

# --- Helper Functions ---

def get_llm(temperature: float = 0.6, thinking_mode: bool = True):
    """
    Get configured LLM instance with Qwen3 optimal parameters.
    
    Qwen3 recommended settings (from HuggingFace):
    - Thinking mode: temperature=0.6, top_p=0.95, top_k=20
    - Non-thinking mode: temperature=0.7, top_p=0.8
    - DO NOT use greedy decoding (causes repetition)
    
    NOTE: vLLM with --enable-reasoning already uses Qwen3 defaults:
    {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}
    """
    # vLLM handles top_k internally when started with --enable-reasoning
    # Just set temperature and top_p
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
    
    # Remove markdown artifacts if present
    content = content.replace("```", "").strip()
    
    # If "Related Work" header exists, extract from there
    if "Related Work" in content:
        # Find the start of actual content
        for marker in ["**Related Work**", "## Related Work", "Related Work\n"]:
            if marker in content:
                parts = content.split(marker, 1)
                if len(parts) > 1:
                    content = parts[1].strip()
                    break
    
    return content.strip()

# --- Nodes ---

def retrieve_papers(state: AgentState):
    """Retrieves papers from Arxiv with (Author, Year) format."""
    print(f"--- Retrieving papers for: {state['topic']} ---")
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=state['topic'],
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers_data = []
    try:
        for result in client.results(search):
            # Format authors for (Author, Year) citation
            authors = [a.name for a in result.authors]
            if len(authors) == 1:
                author_cite = authors[0].split()[-1]  # Last name
            elif len(authors) == 2:
                author_cite = f"{authors[0].split()[-1]} and {authors[1].split()[-1]}"
            else:
                author_cite = f"{authors[0].split()[-1]} et al."
            
            year = result.published.year
            
            paper_info = f"[{author_cite}, {year}]\nTitle: {result.title}\nAuthors: {', '.join(authors)}\nYear: {year}\nAbstract: {result.summary}\n"
            papers_data.append(paper_info)
    except Exception as e:
        print(f"Error fetching from arxiv: {e}")
        
    return {"papers": papers_data}

def generate_related_work(state: AgentState):
    """Generates the Related Work section with (Author, Year) citations."""
    print("--- Generating Related Work Section ---")
    
    if not state['papers']:
        return {"draft": "No papers found to generate a related work section."}

    papers_text = "\n\n".join(state['papers'])
    
    # Build few-shot examples if available
    examples_text = ""
    if state.get('examples'):
        examples_text = "\n\n### Reference Examples (mimic this style):\n"
        for i, ex in enumerate(state['examples'][:3]):  # Limit to 3 for context
            examples_text += f"\n**Example {i+1}:**\n"
            examples_text += f"Papers: {ex.get('source_papers', '')[:600]}...\n"
            examples_text += f"Output: {ex.get('related_work', '')[:800]}...\n"
            examples_text += "-" * 30 + "\n"

    system_prompt = f"""You are an expert academic writer specializing in literature reviews.

### Task
Write a "Related Work" section that synthesizes the provided papers.

### Critical Requirements
1. **Citation Format**: Use (Author, Year) or (Author et al., Year) format. Example: (Smith, 2020), (Jones et al., 2019).
2. **Synthesis**: Group papers by theme, don't just list them sequentially.
3. **Grounding**: Every factual claim must have a citation.
4. **Academic Style**: Formal tone, no first person, smooth transitions.
5. **Gap Analysis**: End with what's missing in the literature.

### Output Format
Write 2-4 paragraphs of continuous prose. No bullet points, no headers.
{examples_text}
"""

    user_prompt = f"""### Topic
{state['topic']}

### Source Papers
{papers_text}

### Instructions
Write the Related Work section now. Use (Author, Year) citations matching the papers above.
Start directly with the content (no "Related Work" header needed)."""

    # Use Qwen3 thinking mode (temperature=0.6) for complex reasoning
    llm = get_llm(temperature=0.6, thinking_mode=True)
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = clean_output(response.content)
        return {"draft": content}
    except Exception as e:
        return {"draft": f"Error: {str(e)}"}

def self_critique(state: AgentState):
    """Self-critique and refine the draft."""
    if not state.get('enable_refinement', False):
        return {"refined_draft": state.get('draft', '')}
    
    print("--- Running Self-Critique ---")
    
    draft = state.get('draft', '')
    papers_text = "\n\n".join(state['papers'])
    
    critique_prompt = f"""You are a strict academic reviewer. Analyze this Related Work draft and identify issues.

### Draft to Review
{draft}

### Source Papers (for fact-checking)
{papers_text[:3000]}

### Check for:
1. Missing citations (claims without (Author, Year))
2. Incorrect attributions
3. Poor transitions between paragraphs
4. Missing gap analysis at the end
5. Repetitive or vague language

### Output
List 2-3 specific issues, then provide a REVISED version that fixes them.
Format:
ISSUES:
- Issue 1
- Issue 2

REVISED:
[Your improved text here]"""

    # Slightly lower temperature for focused editing
    llm = get_llm(temperature=0.5, thinking_mode=True)
    
    try:
        response = llm.invoke([
            SystemMessage(content="You are a meticulous academic editor."),
            HumanMessage(content=critique_prompt)
        ])
        
        content = clean_output(response.content)
        
        # Extract revised section
        if "REVISED:" in content:
            revised = content.split("REVISED:", 1)[1].strip()
        else:
            # If no clear structure, use the whole response as improvement
            revised = content
        
        # Only use revision if it's substantial
        if len(revised) > 100:
            return {"refined_draft": revised}
        else:
            return {"refined_draft": draft}
            
    except Exception as e:
        print(f"Critique error: {e}")
        return {"refined_draft": draft}

def finalize(state: AgentState):
    """Finalize the output."""
    # Use refined draft if available, otherwise use original draft
    final = state.get('refined_draft') or state.get('draft', '')
    return {"draft": final}

# --- Graph Construction ---

def build_workflow(enable_refinement: bool = True):
    """Build the agent workflow graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_papers)
    workflow.add_node("generate", generate_related_work)
    workflow.add_node("critique", self_critique)
    workflow.add_node("finalize", finalize)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "critique")
    workflow.add_edge("critique", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# Default app without refinement (for benchmark compatibility)
app = build_workflow(enable_refinement=False)

# App with refinement
app_with_refinement = build_workflow(enable_refinement=True)

# --- Standalone Generation Function (for benchmarks) ---

def generate_related_work_v2(state: dict, enable_refinement: bool = False) -> dict:
    """
    Standalone generation function for benchmark compatibility.
    Takes a state dict and returns updated state with 'draft'.
    Optimized for 16K context window.
    
    Now uses:
    - target_abstract: What the paper is about
    - fields_of_study: Thematic hints for grouping
    """
    print("--- Generating Related Work Section (v2) ---")
    
    if not state.get('papers'):
        return {"draft": "No papers found."}

    # Use more context with 16K window
    papers_text = "\n\n".join(state['papers'])[:MAX_PAPERS_CONTEXT]
    
    # NEW: Extract additional context if available
    target_abstract = state.get('target_abstract', '')
    fields_of_study = state.get('fields_of_study', '')
    
    # Build context about the target paper
    target_context = ""
    if target_abstract:
        target_context += f"\n### About This Paper\n{target_abstract[:1000]}\n"
    if fields_of_study:
        target_context += f"\n### Research Fields\n{fields_of_study}\n"
    
    # Build examples text with more content
    examples_text = ""
    if state.get('examples'):
        examples_text = "\n\n### Reference Examples (study these carefully):\n"
        remaining_budget = MAX_EXAMPLES_CONTEXT
        for i, ex in enumerate(state['examples'][:5]):  # Up to 5 examples now
            if remaining_budget <= 0:
                break
            ex_text = f"\nExample {i+1}:\n"
            ex_text += f"Papers: {ex.get('source_papers', '')[:600]}...\n"
            ex_text += f"Output: {ex.get('related_work', '')[:800]}...\n"
            if len(ex_text) <= remaining_budget:
                examples_text += ex_text
                remaining_budget -= len(ex_text)

    system_prompt = f"""You are an expert academic writer. Write a Related Work section.

### Requirements
- Use (Author, Year) or (Author et al., Year) citations
- Synthesize papers thematically (don't list sequentially)
- Every claim needs a citation
- End with a gap analysis
- 2-4 paragraphs, no headers, no bullet points
- Connect the cited works to the paper's main topic
{examples_text}"""

    user_prompt = f"""### Paper Title
{state['topic']}
{target_context}
### Cited Papers to Synthesize
{papers_text}

### Task
Write a Related Work section that synthesizes the above papers in the context of this paper's research.
Focus on how these works relate to the paper's topic and identify gaps they don't address."""

    # Use Qwen3 thinking mode for complex synthesis
    llm = get_llm(temperature=0.6, thinking_mode=True)
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        draft = clean_output(response.content)
        
        # Optional refinement (more context with 16K window)
        if enable_refinement and len(draft) > 50:
            print("--- Running Refinement ---")
            refine_prompt = f"""Improve this Related Work section. Check for and fix:
1. Missing citations - every claim needs (Author, Year)
2. Poor transitions between paragraphs
3. Vague or unsupported statements
4. Missing gap analysis at the end
5. Repetitive phrasing

Current draft:
{draft}

Source papers (for fact-checking and adding citations):
{papers_text[:MAX_REFINEMENT_CONTEXT]}

Output ONLY the improved text (no explanation, no preamble):"""
            
            response2 = llm.invoke([
                SystemMessage(content="You are a meticulous academic editor. Output only the improved Related Work text."),
                HumanMessage(content=refine_prompt)
            ])
            refined = clean_output(response2.content)
            if len(refined) > 100:
                draft = refined
        
        return {"draft": draft}
        
    except Exception as e:
        return {"draft": f"Error: {str(e)}"}

# --- Main ---

if __name__ == "__main__":
    print("Single Agent v2 - Improved Related Work Generator")
    print("=" * 50)
    topic = input("Enter the research topic: ")
    
    if topic:
        initial_state = {
            "topic": topic,
            "papers": [],
            "draft": None,
            "refined_draft": None,
            "examples": None,
            "enable_refinement": True
        }
        result = app_with_refinement.invoke(initial_state)
        
        print("\n\n=== Generated Related Work ===\n")
        print(result.get("draft"))
    else:
        print("No topic entered.")

