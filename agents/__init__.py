"""
Multi-Agent System for Related Work Generation

Agents:
- PlannerAgent: Theme extraction + retrieval + workflow routing
- OutlinerAgent: Section structure + citation anchors
- WriterAgent: Paragraph generation with scientific tone
- CriticAgent: Quality control + hallucination detection
"""

from agents.base import get_llm, clean_output, MASConfig, extract_thinking, extract_key_terms
from agents.planner import PlannerAgent
from agents.outliner import OutlinerAgent
from agents.writer import WriterAgent
from agents.critic import CriticAgent

__all__ = [
    'get_llm',
    'clean_output',
    'extract_thinking',
    'extract_key_terms',
    'MASConfig',
    'PlannerAgent',
    'OutlinerAgent',
    'WriterAgent',
    'CriticAgent'
]
