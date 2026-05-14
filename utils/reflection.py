import json
import os
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

async def call_llm_with_reflection(system_prompt: str, user_prompt: str, response_model: BaseModel, model: str = "gpt-4o") -> BaseModel:
    """
    Calls the LLM to generate structured output matching the provided Pydantic model.
    Implements a Reflect Loop where the model self-corrects based on theoretical constraints.
    """
    
    # Step 1: Initial Generation
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=response_model,
        temperature=0.3
    )
    
    initial_output = response.choices[0].message.parsed
    
    # Step 2: Reflect Loop (Self-Correction)
    # Add Orchestrator-specific checks if this is Agent 5
    ratio_check = ""
    if response_model.__name__ == "Agent5Output":
        ratio_check = """
    5. CRITICAL ORCHESTRATOR CHECK: Verify the Evidence Ratios. 
       - Executive Summary must be 70% qualitative / 30% quantitative.
       - Pragmatic Analysis must be 55% qualitative / 45% quantitative.
       - Semantic/Register must be 60% qualitative / 40% quantitative.
       - Corpus Statistics must be 90% quantitative.
       If any of these sections lack sufficient quantitative (numbers, frequencies, collocations) or qualitative (segment citations, theory) evidence, YOU MUST REWRITE THAT SECTION to meet the exact ratio.
        """

    reflection_prompt = f"""
    You are an expert computational linguist verifying the output of a subordinate AI agent.
    Review the following structured JSON output for theoretical consistency and accuracy.
    
    Initial Output:
    {initial_output.model_dump_json(indent=2)}
    
    Tasks:
    1. If a Speech Act is classified as 'Directive', ensure the text actually contains a command, request, or imperative.
    2. If Gricean maxims are violated, ensure the explanation makes logical sense.
    3. If there are contradictions between quantitative and qualitative claims, correct them.
    4. If it is fully correct, return the identical JSON structure.
    {ratio_check}
    
    Return the corrected JSON output.
    """
    
    correction_response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Self-Correction Reflection Agent."},
            {"role": "user", "content": reflection_prompt}
        ],
        response_format=response_model,
        temperature=0.1
    )
    
    return correction_response.choices[0].message.parsed
