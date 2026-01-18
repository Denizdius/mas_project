"""
MAS Workflow Orchestrator using LangGraph.

Implements the complete pipeline:
USER INPUT → PLANNER → OUTLINER → WRITER → CRITIC → ACCEPT? → FINAL ASSEMBLY

Features:
- Feedback loop for revision (max 5 iterations)
- Parallel section writing
- Final assembly with smooth transitions
"""

from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
import operator

from agents.base import MASConfig, get_llm, clean_output
from agents.planner import PlannerAgent
from agents.outliner import OutlinerAgent
from agents.writer import WriterAgent, WriterOutput
from agents.critic import CriticAgent, CriticOutput
from langchain_core.messages import HumanMessage, SystemMessage


# --- State Definition ---

class SectionState(TypedDict):
    """State for a single section in the pipeline."""
    section: Dict  # SectionOutline from Outliner
    writer_output: Optional[Dict]  # WriterOutput as dict
    critic_output: Optional[Dict]  # CriticOutput as dict
    revision_count: int
    accepted: bool


class MASState(TypedDict):
    """Global state for the Multi-Agent System."""
    # Input
    topic: str
    target_abstract: str
    fields_of_study: str
    provided_papers: List[Dict]  # For benchmark mode
    
    # Pipeline outputs
    planner_output: Optional[Dict]
    outliner_output: Optional[Dict]
    section_states: List[SectionState]
    
    # Final output
    final_text: str
    
    # Config
    config: Dict


class MASWorkflow:
    """
    Multi-Agent System Workflow Orchestrator.
    
    Coordinates:
    - PlannerAgent: Theme extraction + retrieval
    - OutlinerAgent: Section structure
    - WriterAgent: Paragraph generation
    - CriticAgent: Quality control
    - Final Assembly: Combine all accepted sections
    """
    
    def __init__(self, config: MASConfig = None):
        self.config = config or MASConfig()
        
        # Initialize agents
        self.planner = PlannerAgent(self.config)
        self.outliner = OutlinerAgent(self.config)
        self.writer = WriterAgent(self.config)
        self.critic = CriticAgent(self.config)
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        workflow = StateGraph(MASState)
        
        # Add nodes
        workflow.add_node("planner", self._run_planner)
        workflow.add_node("outliner", self._run_outliner)
        workflow.add_node("writer", self._run_writer)
        workflow.add_node("critic", self._run_critic)
        workflow.add_node("revision_router", self._revision_router)
        workflow.add_node("final_assembly", self._final_assembly)
        
        # Define edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "outliner")
        workflow.add_edge("outliner", "writer")
        workflow.add_edge("writer", "critic")
        workflow.add_edge("critic", "revision_router")
        
        # Conditional edge: continue revision or go to final assembly
        workflow.add_conditional_edges(
            "revision_router",
            self._should_continue_revision,
            {
                "revise": "writer",
                "done": "final_assembly"
            }
        )
        
        workflow.add_edge("final_assembly", END)
        
        return workflow.compile()
    
    def _run_planner(self, state: MASState) -> Dict:
        """Run Planner Agent."""
        print("\n" + "="*60)
        print("STAGE 1: PLANNER AGENT")
        print("="*60)
        
        result = self.planner.run(
            topic=state['topic'],
            target_abstract=state.get('target_abstract', ''),
            fields_of_study=state.get('fields_of_study', ''),
            provided_papers=state.get('provided_papers', None)
        )
        
        return {"planner_output": result}
    
    def _run_outliner(self, state: MASState) -> Dict:
        """Run Outliner Agent."""
        print("\n" + "="*60)
        print("STAGE 2: OUTLINER AGENT")
        print("="*60)
        
        planner_output = state.get('planner_output', {})
        
        if not planner_output or not planner_output.get('themes'):
            return {
                "outliner_output": {"sections": [], "gap_analysis_hint": "No themes found"},
                "section_states": []
            }
        
        result = self.outliner.run(planner_output)
        
        # Initialize section states
        section_states = []
        for section in result.get('sections', []):
            section_states.append({
                "section": section,
                "writer_output": None,
                "critic_output": None,
                "revision_count": 0,
                "accepted": False
            })
        
        return {
            "outliner_output": result,
            "section_states": section_states
        }
    
    def _run_writer(self, state: MASState) -> Dict:
        """Run Writer Agent for all sections that need writing/revision."""
        print("\n" + "="*60)
        print("STAGE 3: WRITER AGENT")
        print("="*60)
        
        topic = state['topic']
        section_states = state.get('section_states', [])
        
        if not section_states:
            return {"section_states": []}
        
        updated_states = []
        sections_to_write = []
        
        # Identify sections needing writing
        for i, ss in enumerate(section_states):
            if not ss['accepted']:
                # Check if this is a revision
                if ss.get('critic_output') and not ss['accepted']:
                    feedback = ss['critic_output'].get('feedback', '')
                else:
                    feedback = None
                
                sections_to_write.append((i, ss['section'], feedback))
            updated_states.append(dict(ss))
        
        # Write sections (parallel if enabled)
        if sections_to_write:
            num_sections = len(section_states)
            
            for idx, section, feedback in sections_to_write:
                is_last = (idx == num_sections - 1)
                
                if feedback:
                    # Revision
                    prev_output = updated_states[idx].get('writer_output')
                    if prev_output:
                        prev_writer_output = WriterOutput(
                            section_id=prev_output.get('section_id', idx),
                            text=prev_output.get('text', ''),
                            citations_used=prev_output.get('citations_used', []),
                            revision_count=prev_output.get('revision_count', 0)
                        )
                        result = self.writer.revise_section(
                            section, topic, prev_writer_output, feedback, is_last
                        )
                    else:
                        result = self.writer.write_section(section, topic, is_last, feedback)
                else:
                    result = self.writer.write_section(section, topic, is_last)
                
                updated_states[idx]['writer_output'] = result.to_dict()
                updated_states[idx]['revision_count'] = result.revision_count
        
        return {"section_states": updated_states}
    
    def _run_critic(self, state: MASState) -> Dict:
        """Run Critic Agent to evaluate all written sections."""
        print("\n" + "="*60)
        print("STAGE 4: CRITIC AGENT")
        print("="*60)
        
        section_states = state.get('section_states', [])
        
        if not section_states:
            return {"section_states": []}
        
        updated_states = []
        
        for ss in section_states:
            ss_copy = dict(ss)
            
            # Skip already accepted sections
            if ss.get('accepted'):
                updated_states.append(ss_copy)
                continue
            
            writer_output = ss.get('writer_output')
            if not writer_output:
                updated_states.append(ss_copy)
                continue
            
            # Evaluate
            critic_result = self.critic.evaluate(writer_output, ss['section'])
            ss_copy['critic_output'] = critic_result.to_dict()
            ss_copy['accepted'] = critic_result.accept
            
            updated_states.append(ss_copy)
        
        return {"section_states": updated_states}
    
    def _revision_router(self, state: MASState) -> Dict:
        """Router node - doesn't modify state, just passes through."""
        return {}
    
    def _should_continue_revision(self, state: MASState) -> str:
        """
        Decide whether to continue revision or proceed to final assembly.
        
        Returns "revise" if any section needs revision and under max limit.
        Returns "done" if all accepted or max revisions reached.
        """
        section_states = state.get('section_states', [])
        max_revisions = self.config.max_revisions
        
        needs_revision = False
        
        for ss in section_states:
            if not ss.get('accepted'):
                if ss.get('revision_count', 0) < max_revisions:
                    needs_revision = True
                    print(f"[ROUTER] Section {ss['section'].get('section_id', '?')} needs revision "
                          f"(attempt {ss.get('revision_count', 0) + 1}/{max_revisions})")
                else:
                    print(f"[ROUTER] Section {ss['section'].get('section_id', '?')} max revisions reached, force accepting")
        
        if needs_revision:
            return "revise"
        else:
            print("[ROUTER] All sections accepted or max revisions reached → Final Assembly")
            return "done"
    
    def _final_assembly(self, state: MASState) -> Dict:
        """
        Final Assembly: Combine all sections into cohesive output.
        
        Uses Planner Agent role to:
        - Combine all accepted sections
        - Ensure smooth transitions
        - Add final formatting
        """
        print("\n" + "="*60)
        print("STAGE 5: FINAL ASSEMBLY")
        print("="*60)
        
        section_states = state.get('section_states', [])
        outliner_output = state.get('outliner_output', {})
        gap_hint = outliner_output.get('gap_analysis_hint', '')
        
        if not section_states:
            return {"final_text": "No content generated."}
        
        # Collect all section texts
        section_texts = []
        for ss in section_states:
            writer_output = ss.get('writer_output', {})
            text = writer_output.get('text', '')
            if text:
                section_texts.append(text)
        
        if not section_texts:
            return {"final_text": "No content generated."}
        
        # If only one section, return it directly
        if len(section_texts) == 1:
            print("[ASSEMBLY] Single section - returning directly")
            return {"final_text": section_texts[0]}
        
        # Multiple sections - combine and smooth transitions
        combined = "\n\n".join(section_texts)
        
        # Use LLM to smooth transitions
        smoothed = self._smooth_transitions(combined, state['topic'])
        
        print(f"[ASSEMBLY] Combined {len(section_texts)} sections into final output")
        
        return {"final_text": smoothed}
    
    def _smooth_transitions(self, combined_text: str, topic: str) -> str:
        """
        Use LLM to smooth transitions between paragraphs.
        """
        llm = get_llm(temperature=0.4, thinking_mode=True)
        
        prompt = f"""You are an expert academic editor. Your task is to LIGHTLY EDIT the following 
Related Work section to ensure smooth transitions between paragraphs.

### Rules:
1. DO NOT change any factual content or citations
2. DO NOT remove or add citations
3. ONLY improve transitions between paragraphs
4. Keep the academic tone
5. Output ONLY the edited text (no preamble)

### Topic: {topic}

### Text to Edit:
{combined_text}

### Edited Text:"""

        try:
            response = llm.invoke([
                SystemMessage(content="You are a meticulous academic editor. Make minimal changes - only improve transitions."),
                HumanMessage(content=prompt)
            ])
            
            result = clean_output(response.content)
            
            # Sanity check - don't use if too different in length
            if 0.7 < len(result) / len(combined_text) < 1.3:
                return result
            else:
                print("[ASSEMBLY] Smoothing changed too much, using original")
                return combined_text
                
        except Exception as e:
            print(f"[ASSEMBLY] Smoothing error: {e}, using original")
            return combined_text
    
    def run(self, topic: str, target_abstract: str = "", fields_of_study: str = "",
            provided_papers: List[Dict] = None) -> str:
        """
        Run the complete MAS workflow.
        
        Args:
            topic: Paper title
            target_abstract: Abstract of target paper
            fields_of_study: Research fields
            provided_papers: Papers for benchmark mode (each with citation_key, title, authors, year, abstract, full_text)
        
        Returns:
            Final Related Work section text
        """
        initial_state: MASState = {
            "topic": topic,
            "target_abstract": target_abstract,
            "fields_of_study": fields_of_study,
            "provided_papers": provided_papers or [],
            "planner_output": None,
            "outliner_output": None,
            "section_states": [],
            "final_text": "",
            "config": {
                "max_revisions": self.config.max_revisions,
                "retrieval_mode": self.config.retrieval_mode
            }
        }
        
        print("\n" + "#"*60)
        print("# MULTI-AGENT SYSTEM: Related Work Generation")
        print("#"*60)
        print(f"Topic: {topic[:60]}...")
        print(f"Mode: {self.config.retrieval_mode}")
        print(f"Max revisions: {self.config.max_revisions}")
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        print("\n" + "#"*60)
        print("# WORKFLOW COMPLETE")
        print("#"*60)
        
        return final_state.get('final_text', '')


# --- Convenience function ---

def run_mas(topic: str, 
            target_abstract: str = "",
            fields_of_study: str = "",
            provided_papers: List[Dict] = None,
            config: MASConfig = None) -> str:
    """
    Convenience function to run MAS workflow.
    
    Args:
        topic: Paper title
        target_abstract: Abstract of target paper
        fields_of_study: Research fields
        provided_papers: Papers for benchmark mode
        config: MASConfig object
    
    Returns:
        Generated Related Work section
    """
    workflow = MASWorkflow(config)
    return workflow.run(topic, target_abstract, fields_of_study, provided_papers)


# --- Main ---

if __name__ == "__main__":
    print("Testing MAS Workflow...")
    
    # Test with live Arxiv retrieval
    config = MASConfig()
    config.retrieval_mode = "live"
    config.max_revisions = 2  # Limit for testing
    
    result = run_mas(
        topic="Transformer models for natural language processing",
        target_abstract="We propose a new efficient transformer architecture...",
        config=config
    )
    
    print("\n\n" + "="*60)
    print("FINAL OUTPUT")
    print("="*60)
    print(result)

