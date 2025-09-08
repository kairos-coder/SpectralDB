# server.py
"""
The Sacred Conduit: Enhanced with divine resilience and offline synthesis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import json
import random
import asyncio
from datetime import datetime
from pathlib import Path
import re

# Import our core components
from spectraldb import SpectralDB
from playwright_conductor import PlaywrightConductor, TARGET_REGISTRY

app = FastAPI(title="Orchestral Conduit", description="The bridge between Telos and SpectralDb")

# Mount the frontend (the Micro-Synthesizer)
app.mount("/static", StaticFiles(directory="."), name="static")

# Initialize the Heart
db = SpectralDB("orchestral_memory.db")

# Pydantic models for structured data
class IngestionRequest(BaseModel):
    input: str
    atoms: List[str]
    variations: List[str]
    timestamp: str

class SynthesisRequest(BaseModel):
    user_prompt: str
    target_oracle: str = "claude_ai"  # Default oracle

# --- Divine Enhancements ---

def sanitize_input(text: str) -> str:
    """Basic sanitization to protect against HTML/script injection."""
    if not text:
        return text
    # Remove HTML tags
    cleaned = re.sub(r'<[^>]*>', '', text)
    # Escape SQL special characters (simple version for display safety)
    cleaned = cleaned.replace("'", "''")
    return cleaned

async def async_ingest_memory(db_instance, *args, **kwargs):
    """Run SpectralDB ingestion in a thread to avoid blocking the async loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, db_instance.ingest_memory, *args, **kwargs)

def generate_offline_synthesis(user_prompt: str, variations_count: int = 5) -> str:
    """The Micro-Synthesizer's core logic as a fallback oracle."""
    atoms = user_prompt.split()
    if not atoms:
        return "The input was silent. No atoms to synthesize."
    
    # Create varied permutations
    synthesized_variations = []
    for i in range(variations_count):
        shuffled = random.sample(atoms, len(atoms))
        synthesized_variations.append(" ".join(shuffled))
    
    # Validate and join
    validated = [v.capitalize().rstrip('.') + '.' for v in synthesized_variations]
    return "\n".join(validated)

# --- API Endpoints ---

@app.get("/")
async def serve_frontend():
    """Serve the Micro-Synthesizer frontend"""
    return FileResponse("index.html")

@app.post("/api/ingest")
async def ingest_microsynthesis(request: IngestionRequest):
    """
    Ingest the output of the Micro-Synthesizer into SpectralDb.
    Establishes lineage: User Input -> Atoms -> Variations.
    """
    try:
        # Sanitize input
        sanitized_input = sanitize_input(request.input)
        sanitized_atoms = [sanitize_input(atom) for atom in request.atoms]
        sanitized_variations = [sanitize_input(v) for v in request.variations]

        # Ingest the original user input as the root memory (async)
        root_memory_id = await async_ingest_memory(
            db, sanitized_input, "microsynthesizer", ["user_input", "root"]
        )

        # Ingest each atom as a child of the root (async)
        atom_tasks = [
            async_ingest_memory(
                db, atom, "microsynthesizer", ["atom", "primitive"], [root_memory_id], "atomic_deconstruction"
            ) for atom in sanitized_atoms
        ]
        atom_memory_ids = await asyncio.gather(*atom_tasks)

        # Ingest each variation as a child of the root (async)
        variation_tasks = [
            async_ingest_memory(
                db, variation, "microsynthesizer", ["variation", "synthesis"], [root_memory_id], "combinatorial_creation"
            ) for variation in sanitized_variations
        ]
        variation_memory_ids = await asyncio.gather(*variation_tasks)

        return JSONResponse({
            "status": "success",
            "root_memory_id": root_memory_id,
            "atom_memory_ids": atom_memory_ids,
            "variation_memory_ids": variation_memory_ids,
            "message": "Micro-synthesis ingested into SpectralDb. Lineage established."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/api/orchestrate")
async def orchestrate_synthesis(request: SynthesisRequest):
    """
    The full Orchestral ritual: User input -> SpectralDb -> Phantom Knight -> SpectralDb.
    Now with divine offline fallback.
    """
    try:
        # Sanitize the user's prompt
        sanitized_prompt = sanitize_input(request.user_prompt)
        
        # 1. Ingest the user's original prompt into the Heart. (async)
        root_memory_id = await async_ingest_memory(
            db, sanitized_prompt, "user", ["orchestral_ritual", "root_intent"]
        )

        prophecy = None
        source = request.target_oracle

        # 2. Try to dispatch the Phantom Page Knight to the oracle.
        try:
            async with PlaywrightConductor(headless=True) as knight:
                prophecy = await knight.parley_with_oracle(
                    TARGET_REGISTRY[request.target_oracle], 
                    sanitized_prompt,
                    max_retries=1
                )
        except Exception as e:
            logger.warning(f"Oracle quest failed: {e}. Falling back to offline synthesis.")

        # 3. FALLBACK: If the oracle is silent, use the Micro-Synthesizer.
        if not prophecy:
            logger.warning("Oracle unreachable. Generating local synthesis...")
            prophecy = generate_offline_synthesis(sanitized_prompt)
            source = "offline_microsynthesizer"  # Change source to reflect fallback

        # 4. Ingest the prophecy (async)
        prophecy_memory_id = await async_ingest_memory(
            db, prophecy, source, ["response", "synthesis"],
            [root_memory_id], "phantom_page_quest", "perfect_fifth"
        )

        return JSONResponse({
            "status": "success",
            "source": source,
            "root_memory_id": root_memory_id,
            "prophecy_memory_id": prophecy_memory_id,
            "prophecy_preview": prophecy[:100] + "..." if len(prophecy) > 100 else prophecy,
            "message": "Orchestral ritual complete. Knowledge has a past."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")

@app.get("/api/memories")
async def get_memories(limit: int = 10, tag: Optional[str] = None, source: Optional[str] = None):
    """Get recent memories from SpectralDb, with filtering."""
    # For now, we get all and filter in-memory. For large scales, push to SQL.
    memories = db.get_recent_memories(limit=100)  # Get extra to filter
    filtered_memories = memories
    
    if tag:
        filtered_memories = [m for m in filtered_memories if tag in json.loads(m.get('tags', '[]'))]
    if source:
        filtered_memories = [m for m in filtered_memories if m.get('source') == source]
    
    return JSONResponse({"memories": filtered_memories[:limit]})

@app.get("/api/stats")
async def get_stats():
    """Get database statistics"""
    stats = db.get_database_stats()
    return JSONResponse(stats)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
