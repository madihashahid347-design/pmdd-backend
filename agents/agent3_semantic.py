from typing import List
from schemas.agent_schemas import Agent3Output, SegmentData
from utils.reflection import call_llm_with_reflection

def get_adaptive_system_prompt(register_context: str) -> str:
    """Adapts the linguistic framework based on the corpus register."""
    if "informal" in register_context.lower() or "social media" in register_context.lower():
        framework_note = "Use sociolinguistic register analysis (Labov) suitable for informal/colloquial text."
    else:
        framework_note = "Use standard Hallidayan Register Analysis (Field, Tenor, Mode) for formal text."

    return f"""You are a computational linguistics expert focusing on Semantic Field Theory and Register Analysis.
{framework_note}

Analyze the given text segments focusing on target keywords.
1. Group keywords into semantic fields (e.g., CONFLICT, ECONOMY, SOLIDARITY).
2. Classify the register exactly as Formal, Semi-formal, Informal, Technical, Colloquial, or Unknown.
3. Identify any register borrowing (e.g., formal terms used in an informal context).
4. Provide a brief discourse shift analysis and any general notes.
"""

async def run_agent3(segments: List[SegmentData], keywords: List[str], register_context: str = "general") -> Agent3Output:
    """Analyzes semantic fields and register for the corpus containing the keywords."""
    system_prompt = get_adaptive_system_prompt(register_context)
    
    # Filter segments that contain the target keywords
    relevant_segments = []
    for seg in segments:
        text_lower = seg.text.lower()
        if any(kw.lower() in text_lower for kw in keywords):
            relevant_segments.append(seg.text)
            
    if not relevant_segments:
        return Agent3Output(
            semantic_fields=[],
            register="Unknown",
            register_borrowing_detected=False,
            discourse_shift_analysis="No target keywords found in corpus.",
            notes="Analysis skipped."
        )
        
    # Analyze a chunk of relevant segments to save context window
    sample_text = " | ".join(relevant_segments[:20])
    user_prompt = f"Target Keywords: {', '.join(keywords)}\nText Sample: {sample_text}"
    
    try:
        # call_llm_with_reflection enforces the Agent3Output Pydantic schema
        analysis: Agent3Output = await call_llm_with_reflection(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=Agent3Output
        )
        return analysis
    except Exception as e:
        print(f"Error in Agent 3: {e}")
        return Agent3Output(
            semantic_fields=[],
            register="Unknown",
            register_borrowing_detected=False,
            discourse_shift_analysis="Error during analysis.",
            notes=str(e)
        )
