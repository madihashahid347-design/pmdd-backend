import json
from typing import List
from schemas.agent_schemas import Agent1Output, Agent2SegmentOutput, Agent3Output, Agent4Output, Agent5Output
from utils.reflection import call_llm_with_reflection
from database.client import retrieve_episodic_memory

async def run_orchestrator(
    a1_out: Agent1Output, 
    a2_out: List[Agent2SegmentOutput], 
    a3_out: Agent3Output, 
    a4_out: Agent4Output, 
    keywords: List[str]
) -> Agent5Output:
    """Synthesizes evidence and generates the final scientific report using structured outputs."""
    
    # 1. Retrieve Episodic Memory from Supabase
    past_memory = retrieve_episodic_memory(keywords)
    memory_str = json.dumps(past_memory, indent=2) if past_memory else "No prior memory for these keywords."
    
    # 2. Prepare Synthesis Prompt
    system_prompt = """You are Agent 5, the Orchestrator and Evidence Synthesizer.
You must generate a formal, scientific linguistic evidence report suitable for a research community.

CRITICAL EVIDENCE RATIO REQUIREMENTS:
You MUST adhere to the following evidence ratios across your sections:
- Executive Summary: 70% qualitative / 30% quantitative
- Pragmatic Drift Evidence: 55% qualitative / 45% quantitative
- Semantic Field & Register Analysis: 60% qualitative / 40% quantitative
- Quantitative Corpus Statistics: 90% quantitative
- Linguistic Theory Mapping: 85% qualitative

INSTRUCTIONS:
1. Quantitative Evidence: Cite TTR, exact frequencies, and collocation data (from Agent 4).
2. Qualitative Evidence: Cite exact segment IDs, speech acts, and implicatures (from Agent 2 & 3).
3. Theoretical Grounding: Explicitly link findings to linguistic theory (e.g., Grice, Searle, Halliday).
4. Episodic Memory: Compare current findings with the provided 'Historical Memory'.

You MUST return a JSON object that matches the requested schema EXACTLY, filling out all 11 sections.
"""

    # We need to serialize the pydantic models to JSON strings for the prompt
    user_prompt = f"""
--- CORPUS STATS (Agent 1) ---
{a1_out.model_dump_json()}

--- PRAGMATIC SAMPLE (Agent 2) ---
{json.dumps([seg.model_dump() for seg in a2_out[:5]])}

--- SEMANTIC & REGISTER (Agent 3) ---
{a3_out.model_dump_json()}

--- QUANTITATIVE STATS (Agent 4) ---
{a4_out.model_dump_json()}

--- HISTORICAL MEMORY ---
{memory_str}
"""

    try:
        # call_llm_with_reflection enforces the Agent5Output Pydantic schema
        report: Agent5Output = await call_llm_with_reflection(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=Agent5Output
        )
        return report
    except Exception as e:
        print(f"Error in Orchestrator: {e}")
        # Create a dummy failure report matching schema
        from schemas.agent_schemas import MeaningDriftScores
        return Agent5Output(
            executive_summary="Synthesis failed.",
            corpus_profile="",
            pragmatic_drift_evidence="",
            semantic_register_analysis="",
            quantitative_corpus_statistics="",
            multi_agent_synthesis="",
            linguistic_theory_mapping="",
            conclusions=str(e),
            methodology="",
            references="",
            appendix="",
            scores=MeaningDriftScores(overall=0, pragmatic=0, semantic=0, register=0)
        )
