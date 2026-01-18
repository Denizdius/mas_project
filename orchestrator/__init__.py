"""
Orchestrator: LangGraph-based workflow coordination.

Implements the Multi-Agent workflow:
1. Planner → Theme extraction
2. Outliner → Section structure
3. Writer → Paragraph generation (parallel)
4. Critic → Quality check
5. Accept? → Loop back to Writer or proceed
6. Final Assembly → Combine all sections
"""

from orchestrator.workflow import MASWorkflow, run_mas

__all__ = ['MASWorkflow', 'run_mas']

