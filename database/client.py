import os
from typing import Dict, Any, List

def save_episodic_memory(analysis_id: str, keyword: str, semantic_fields: List[Dict], collocations: Dict, pragmatic_trends: Dict, register_context: str):
    """Saves agent findings to Supabase Episodic Memory table (Mocked for local dev)."""
    print(f"Mocked Supabase: Saved memory for keyword '{keyword}' in analysis {analysis_id}")

def retrieve_episodic_memory(keywords: List[str]) -> List[Dict]:
    """Retrieves previous findings to act as historical context for Agents (Mocked for local dev)."""
    print(f"Mocked Supabase: Retrieved memory for keywords {keywords}")
    return []
