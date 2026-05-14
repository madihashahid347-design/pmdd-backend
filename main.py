from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio
import uuid
import time
from typing import List
from agents.agent1_preprocessor import run_agent1
from agents.agent2_pragmatic import run_agent2
from agents.agent3_semantic import run_agent3
from agents.agent4_statistics import run_agent4
from agents.agent5_orchestrator import run_orchestrator
from database.client import save_episodic_memory

app = FastAPI(title="PMDD Platform API", version="1.0.0")

# CORS Configuration for Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://frontend-eta-blue-52.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory mock store for SSE events for active analysis sessions
# In production, this would use Redis Pub/Sub for scalability across worker nodes.
analysis_event_queues = {}

@app.get("/")
async def root():
    return {"message": "PMDD Backend API is running."}

@app.post("/upload")
async def upload_corpus(
    file: UploadFile = File(...),
    frameworks: str = Form("[]") # JSON string of selected frameworks
):
    """Handles corpus upload and initializes an analysis session."""
    analysis_id = str(uuid.uuid4())
    
    # Setup an event queue for this analysis session
    analysis_event_queues[analysis_id] = asyncio.Queue()
    
    # In a real scenario, we would save the file to Supabase Storage, 
    # insert a record into the 'analyses' table, and then dispatch a background task 
    # to process it. For now, we simulate the start.
    
    return {
        "message": "Upload successful",
        "analysis_id": analysis_id,
        "filename": file.filename
    }

async def execute_pipeline(analysis_id: str, file_content: str, filename: str, keywords: List[str], register_context: str = "general"):
    """Executes the full 5-agent pipeline and streams SSE events."""
    queue = analysis_event_queues.get(analysis_id)
    if not queue:
        return
        
    try:
        # Agent 1
        await queue.put({"data": "[Agent 1] Cleaning and segmenting corpus..."})
        a1_out = run_agent1(file_content, filename)
        
        # Chunking Logic for Long Corpora (Agent 1 Output)
        CHUNKS_SIZE = 20 # Max segments per LLM call to save tokens
        total_segments = len(a1_out.segments)
        
        await queue.put({"data": f"[System] Corpus segmented into {total_segments} chunks. Processing in batches..."})
        
        # Agent 2
        await queue.put({"data": "[Agent 2] Running Pragmatic Analysis (Speech Acts, Implicatures)..."})
        a2_out = []
        for i in range(0, min(total_segments, CHUNKS_SIZE * 3), CHUNKS_SIZE):
             batch = a1_out.segments[i:i+CHUNKS_SIZE]
             batch_out = await run_agent2(batch)
             a2_out.extend(batch_out)
             await queue.put({"data": f"[Agent 2] Processed batch {i//CHUNKS_SIZE + 1}..."})
        
        # Agent 3
        await queue.put({"data": "[Agent 3] Mapping Semantic Fields and Register..."})
        a3_out = await run_agent3(a1_out.segments[:CHUNKS_SIZE * 3], keywords, register_context)
        
        # Agent 4
        await queue.put({"data": "[Agent 4] Computing Collocations and Corpus Statistics..."})
        a4_out = run_agent4(a1_out.segments, keywords)
        
        # Agent 5
        await queue.put({"data": "[Agent 5] Synthesizing final scientific report..."})
        a5_out = await run_orchestrator(a1_out, a2_out, a3_out, a4_out, keywords)
        
        # Generate PDF
        await queue.put({"data": "[System] Generating publication-ready PDF..."})
        from services.pdf_generator import generate_scientific_pdf
        pdf_path = generate_scientific_pdf(a5_out, analysis_id)
        
        # Save to episodic memory
        await queue.put({"data": "[System] Saving findings to Episodic Memory..."})
        for kw in keywords:
            save_episodic_memory(
                analysis_id=analysis_id,
                keyword=kw,
                semantic_fields=[sf.model_dump() for sf in a5_out.semantic_fields] if hasattr(a5_out, 'semantic_fields') else [],
                collocations=a4_out.keywords_data.get(kw.lower()).top_collocates if kw.lower() in a4_out.keywords_data else {},
                pragmatic_trends={"overall_pragmatic_score": a5_out.scores.pragmatic},
                register_context=register_context
            )
            
        await queue.put({"data": "COMPLETE"})
        
    except Exception as e:
        await queue.put({"data": f"ERROR: {str(e)}"})
        await queue.put({"data": "COMPLETE"})

@app.post("/analyze/{analysis_id}")
async def start_analysis(analysis_id: str, background_tasks: BackgroundTasks):
    """Triggers the AI agent pipeline for an uploaded corpus."""
    if analysis_id not in analysis_event_queues:
        return {"error": "Invalid analysis ID"}
        
    # In a real app we'd fetch the file content from Supabase storage using the analysis_id.
    # For this implementation, we will pass dummy file content or simulate fetching.
    file_content = "This is a dummy corpus text mentioning power and community."
    keywords = ["power", "community"]
    
    background_tasks.add_task(execute_pipeline, analysis_id, file_content, "dummy.txt", keywords, "general")
    return {"message": "Analysis started"}

@app.get("/stream/{analysis_id}")
async def stream_execution_logs(analysis_id: str):
    """SSE endpoint for streaming real-time agent execution logs."""
    async def event_generator():
        queue = analysis_event_queues.get(analysis_id)
        if not queue:
            yield {"data": "Error: Analysis ID not found or expired"}
            return
            
        while True:
            event = await queue.get()
            if event["data"] == "COMPLETE":
                yield {"data": "Analysis complete!"}
                break
            yield {"event": "message", "data": event["data"]}
            
        # Cleanup
        del analysis_event_queues[analysis_id]

    return EventSourceResponse(event_generator())

# Future endpoints
# @app.get("/report/{id}")
# @app.get("/download/json/{id}")
# @app.get("/download/pdf/{id}")
# @app.get("/dashboard/{id}")

