import spacy
import re
import pandas as pd
from schemas.agent_schemas import Agent1Output, SegmentData, CorpusStats

nlp = None

def get_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            # Fallback if model isn't downloaded yet (for testing)
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load('en_core_web_sm')
    return nlp

def run_agent1(file_content: str, filename: str) -> Agent1Output:
    """Reads a file string (txt or csv), cleans text, segments, and returns validated schema."""
    nlp_model = get_nlp()
    
    raw_text = ""
    if filename.endswith('.csv'):
        # Just extracting lines manually if it's CSV content as a string
        lines = file_content.split('\n')
        # Simple extraction: joining all non-empty lines
        raw_text = " ".join([line.strip() for line in lines if line.strip()])
    else:
        raw_text = file_content
            
    # Clean noise
    text = re.sub(r'\s+', ' ', raw_text).strip()
    
    # Process with spaCy
    doc = nlp_model(text)
    
    segments = []
    total_sentences = len(list(doc.sents))
    
    for i, sent in enumerate(doc.sents):
        word_count = len([t for t in sent if not t.is_space and not t.is_punct])
        if word_count > 0:
            segments.append(SegmentData(
                id=i,
                text=sent.text.strip(),
                word_count=word_count,
                position=round(i / total_sentences, 2) if total_sentences > 0 else 0
            ))
            
    stats = CorpusStats(
        total_sentences=len(segments),
        total_words=sum(s.word_count for s in segments)
    )
            
    return Agent1Output(corpus_stats=stats, segments=segments)
