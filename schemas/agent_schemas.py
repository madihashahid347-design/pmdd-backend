from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

class SegmentData(BaseModel):
    id: int
    text: str
    word_count: int
    position: float

class CorpusStats(BaseModel):
    total_sentences: int
    total_words: int

class Agent1Output(BaseModel):
    corpus_stats: CorpusStats
    segments: List[SegmentData]

class PragmaticAnalysis(BaseModel):
    speech_act: Literal['Assertive', 'Directive', 'Commissive', 'Expressive', 'Declaration', 'Unknown']
    maxim_violations: List[str]
    implicature_type: Literal['conventional', 'conversational', 'none']
    politeness_score: int = Field(ge=1, le=5)
    linguistic_explanation: str
    pragmatic_confidence: float

class Agent2SegmentOutput(SegmentData, PragmaticAnalysis):
    pass

class SemanticFieldData(BaseModel):
    keyword: str
    semantic_field: str

class Agent3Output(BaseModel):
    semantic_fields: List[SemanticFieldData]
    register: Literal['Formal', 'Semi-formal', 'Informal', 'Technical', 'Colloquial', 'Unknown']
    register_borrowing_detected: bool
    discourse_shift_analysis: str
    notes: str

class KeywordStats(BaseModel):
    frequency: int
    relative_frequency_per_10k: float
    top_collocates: Dict[str, int]

class Agent4Output(BaseModel):
    type_token_ratio: float
    total_tokens: int
    keywords_data: Dict[str, KeywordStats]

class MeaningDriftScores(BaseModel):
    overall: float
    pragmatic: float
    semantic: float
    register: float

class Agent5Output(BaseModel):
    executive_summary: str
    corpus_profile: str
    pragmatic_drift_evidence: str
    semantic_register_analysis: str
    quantitative_corpus_statistics: str
    multi_agent_synthesis: str
    linguistic_theory_mapping: str
    conclusions: str
    methodology: str
    references: str
    appendix: str
    scores: MeaningDriftScores
