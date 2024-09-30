# app.py

import os
import asyncio
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from local_search import GraphRAGSearchEngine  # Import the class

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define input data class
class QueryRequest(BaseModel):
    query: str
    search_type: str = Field(default='local', pattern='^(local|global)$')
    max_tokens: Optional[int] = None  # Optional parameter to override default max tokens

# Global variables to store the search engine instances
local_search_engine = None
global_search_engine = None

# Function to initialize the GraphRAGSearchEngine instances
def initialize_search_engines():
    global local_search_engine, global_search_engine

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GRAPHRAG_API_KEY")
    api_base = os.getenv("GRAPHRAG_API_BASE")
    llm_model = os.getenv("GRAPHRAG_LLM_MODEL")
    embedding_model = os.getenv("GRAPHRAG_EMBEDDING_MODEL")

    #NOTE: the input directory when run it in my local machine was ../ but with docker should be ./
    input_dir = "../output/20240930-102622/artifacts"
    lancedb_uri = f"{input_dir}/lancedb"

    # Initialize the local search engine
    logger.info("Initializing local GraphRAGSearchEngine...")
    local_search_engine = GraphRAGSearchEngine(
        input_dir=input_dir,
        lancedb_uri=lancedb_uri,
        api_key=api_key,
        api_base=api_base,
        llm_model=llm_model,
        embedding_model=embedding_model,
        search_type='local',
    )
    logger.info("Local GraphRAGSearchEngine initialized.")

    # Initialize the global search engine
    logger.info("Initializing global GraphRAGSearchEngine...")
    global_search_engine = GraphRAGSearchEngine(
        input_dir=input_dir,
        lancedb_uri=lancedb_uri,
        api_key=api_key,
        api_base=api_base,
        llm_model=llm_model,
        embedding_model=embedding_model,
        search_type='global',
    )
    logger.info("Global GraphRAGSearchEngine initialized.")

# Create the lifespan function
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("Starting up the application...")
    initialize_search_engines()
    yield
    # Shutdown code (if any)
    logger.info("Shutting down the application...")

# Initialize the FastAPI app with the lifespan function
app = FastAPI(lifespan=lifespan)

# Define the query endpoint
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    global local_search_engine, global_search_engine
    if local_search_engine is None or global_search_engine is None:
        raise HTTPException(status_code=500, detail="Search engines not initialized.")

    try:
        logger.info(f"Received query: {request.query}")
        logger.info(f"Search type: {request.search_type}")

        # Select the appropriate search engine
        if request.search_type == 'local':
            search_engine = local_search_engine
        elif request.search_type == 'global':
            search_engine = global_search_engine
        else:
            raise HTTPException(status_code=400, detail="Invalid search_type provided.")

        # Use the run_query_stream method to get an async generator
        async def response_generator():
            async for chunk in search_engine.run_query_stream(request.query):
                yield chunk

        return StreamingResponse(response_generator(), media_type="text/plain")
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with uvicorn if this script is executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
