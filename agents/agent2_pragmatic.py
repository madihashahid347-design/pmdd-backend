from typing import List
import asyncio
from schemas.agent_schemas import SegmentData, Agent2SegmentOutput, PragmaticAnalysis
from utils.reflection import call_llm_with_reflection

SYSTEM_PROMPT = """You are an elite computational linguistics expert.
Analyze the given text segment using these exact frameworks:
1. Speech Act Theory (Searle): Classify as one of the literals provided.
2. Gricean Maxims: List any violations (Quantity, Quality, Relation, Manner).
3. Implicature: Identify as 'conventional', 'conversational', or 'none'.
4. Politeness: Score 1-5 (1=very face-threatening, 5=highly polite).
5. Linguistic Explanation: Provide a 1-sentence theoretical justification.
6. Pragmatic Confidence: 0.0 to 1.0 score representing your certainty.
"""

async def analyze_segment(segment: SegmentData) -> Agent2SegmentOutput:
    user_prompt = f"Analyze this segment: '{segment.text}'"
    
    try:
        # call_llm_with_reflection handles the Pydantic structured output mapping
        analysis: PragmaticAnalysis = await call_llm_with_reflection(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=PragmaticAnalysis
        )
        
        return Agent2SegmentOutput(
            id=segment.id,
            text=segment.text,
            word_count=segment.word_count,
            position=segment.position,
            speech_act=analysis.speech_act,
            maxim_violations=analysis.maxim_violations,
            implicature_type=analysis.implicature_type,
            politeness_score=analysis.politeness_score,
            linguistic_explanation=analysis.linguistic_explanation,
            pragmatic_confidence=analysis.pragmatic_confidence
        )
    except Exception as e:
        print(f"Agent 2 Error on segment {segment.id}: {e}")
        # Fallback
        return Agent2SegmentOutput(
            id=segment.id,
            text=segment.text,
            word_count=segment.word_count,
            position=segment.position,
            speech_act="Unknown",
            maxim_violations=[],
            implicature_type="none",
            politeness_score=3,
            linguistic_explanation="Failed to parse.",
            pragmatic_confidence=0.0
        )

async def run_agent2(segments: List[SegmentData]) -> List[Agent2SegmentOutput]:
    """Runs pragmatic analysis asynchronously on a sample of segments."""
    sample = segments[:20] # Limit for POC to save tokens and time
    
    # Run concurrently
    tasks = [analyze_segment(seg) for seg in sample]
    results = await asyncio.gather(*tasks)
    return list(results)
