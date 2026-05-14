import re
from collections import Counter
from typing import List, Dict
from schemas.agent_schemas import Agent4Output, KeywordStats, SegmentData

def get_collocates(tokens: List[str], target_word: str, window: int = 5) -> Dict[str, int]:
    """Finds words that frequently co-occur with the target word within a window."""
    collocates = Counter()
    target_lower = target_word.lower()
    
    for i, token in enumerate(tokens):
        if token.lower() == target_lower:
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            context = tokens[start:i] + tokens[i+1:end]
            # Simple filter for punctuation and short words
            valid_context = [w.lower() for w in context if len(w) > 2 and w.isalpha()]
            collocates.update(valid_context)
            
    return dict(collocates.most_common(10))

def run_agent4(segments: List[SegmentData], keywords: List[str]) -> Agent4Output:
    """Computes basic corpus statistics, frequencies, and collocations."""
    
    # Extract all tokens from segments
    all_tokens = []
    for seg in segments:
        words = re.findall(r'\b\w+\b', seg.text.lower())
        all_tokens.extend(words)
        
    total_tokens = len(all_tokens)
    unique_types = len(set(all_tokens))
    ttr = round(unique_types / total_tokens, 4) if total_tokens > 0 else 0.0
    
    keywords_data = {}
    
    for kw in keywords:
        kw_lower = kw.lower()
        freq = all_tokens.count(kw_lower)
        collocates = get_collocates(all_tokens, kw_lower, window=5)
        
        rel_freq = round((freq / total_tokens) * 10000, 2) if total_tokens > 0 else 0.0
        
        keywords_data[kw_lower] = KeywordStats(
            frequency=freq,
            relative_frequency_per_10k=rel_freq,
            top_collocates=collocates
        )
        
    return Agent4Output(
        type_token_ratio=ttr,
        total_tokens=total_tokens,
        keywords_data=keywords_data
    )
