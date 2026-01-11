from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Union, Any
import uvicorn
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from fastapi import Request

TICKERS_FILE = "saved_tickers.json"
RESULTS_DIR = "analysis_results"

# Create results directory if it doesn't exist
Path(RESULTS_DIR).mkdir(exist_ok=True)

# Add these helper functions before your endpoints
def load_saved_tickers():
    """Load saved tickers from JSON file"""
    if os.path.exists(TICKERS_FILE):
        try:
            with open(TICKERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_tickers_to_file(tickers_list):
    """Save tickers to JSON file"""
    try:
        with open(TICKERS_FILE, 'w') as f:
            json.dump(tickers_list, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving tickers: {e}")
        return False

def save_analysis_result(ticker, date, decision, config):
    """Save analysis result to JSON file"""
    try:
        timestamp = datetime.now().isoformat()
        filename = f"{RESULTS_DIR}/{ticker}_{date}_{timestamp.replace(':', '-')}.json"
        
        result = {
            "ticker": ticker,
            "date": date,
            "timestamp": timestamp,
            "decision": decision,
            "config": {
                "deep_think_llm": config.get("deep_think_llm"),
                "quick_think_llm": config.get("quick_think_llm"),
                "max_debate_rounds": config.get("max_debate_rounds"),
                "online_tools": config.get("online_tools")
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Analysis saved to: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return None

# Load environment variables from .env file
load_dotenv()

# Verify API keys are loaded
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not found in environment variables")
if not os.getenv("ALPHA_VANTAGE_API_KEY"):
    print("WARNING: ALPHA_VANTAGE_API_KEY not found in environment variables")
else:
    print("✓ API keys loaded successfully")

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

app = FastAPI(title="TradingAgents API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for running blocking operations
executor = ThreadPoolExecutor(max_workers=4)

# Store for async job results
job_results: Dict[str, dict] = {}

class TradingRequest(BaseModel):
    ticker: str
    date: str
    deep_think_llm: Optional[str] = "gpt-4o-mini"
    quick_think_llm: Optional[str] = "gpt-4o-mini"
    max_debate_rounds: Optional[int] = 1
    online_tools: Optional[bool] = False

class TradingResponse(BaseModel):
    ticker: str
    date: str
    decision: Union[Dict[str, Any], str, Any]  # Can be dict, string, or any type
    status: str

DASHBOARD_HTML = """
<!-- Dashboard HTML removed - now loading from external file -->
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <body style="font-family: sans-serif; padding: 40px; text-align: center;">
        <h1>❌ Dashboard Not Found</h1>
        <p>Please create 'dashboard.html' in the same directory as this server.</p>
        <p><a href="/docs">View API Documentation</a></p>
        </body>
        </html>
        """

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze", response_model=TradingResponse)
async def analyze_stock(request: TradingRequest):
    try:
        import subprocess
        import tempfile
        import json
        import sys
        
        print(f"DEBUG: Analyzing {request.ticker} on {request.date}")
        
        # Create a temporary file to pass the configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'ticker': request.ticker,
                'date': request.date,
                'deep_think_llm': request.deep_think_llm,
                'quick_think_llm': request.quick_think_llm,
                'max_debate_rounds': request.max_debate_rounds,
                'online_tools': request.online_tools
            }
            json.dump(config_data, f)
            config_file = f.name
        
        # Create a worker script path
        worker_script = os.path.join(os.path.dirname(__file__), 'trading_worker.py')
        
        # Run analysis in a separate process with UTF-8 encoding
        result = subprocess.run(
            [sys.executable, worker_script, config_file],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=600,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        
        # Clean up temp file
        os.unlink(config_file)
        
        # DEBUG: Print what we got from the worker
        print(f"Worker stdout length: {len(result.stdout)}")
        print(f"Worker stdout (first 500 chars): {result.stdout[:500]}")
        print(f"Worker stderr: {result.stderr[:500] if result.stderr else 'empty'}")
        print(f"Worker return code: {result.returncode}")
        
        if result.returncode != 0:
            print(f"Worker error: {result.stderr}")
            raise Exception(f"Analysis failed: {result.stderr}")
        
        if not result.stdout.strip():
            raise Exception(f"Worker produced no output. stderr: {result.stderr}")
        
        # Parse the result
        try:
            decision = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from worker output")
            print(f"Raw stdout: {result.stdout}")
            raise Exception(f"Worker produced invalid JSON: {str(e)}")
        
        save_analysis_result(request.ticker, request.date, decision, config_data)
        
        return TradingResponse(
            ticker=request.ticker,
            date=request.date,
            decision=decision,
            status="completed"
        )
    except Exception as e:
        print(f"ERROR in analyze_stock: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
async def run_analysis_background(job_id: str, request: TradingRequest):
    try:
        config = DEFAULT_CONFIG.copy()
        config["deep_think_llm"] = request.deep_think_llm
        config["quick_think_llm"] = request.quick_think_llm
        config["max_debate_rounds"] = request.max_debate_rounds
        config["online_tools"] = request.online_tools
        
        ta = TradingAgentsGraph(debug=True, config=config)
        loop = asyncio.get_event_loop()
        _, decision = await loop.run_in_executor(executor, ta.propagate, request.ticker, request.date)
        
        # Convert decision to dict if it's not already
        if not isinstance(decision, dict):
            decision = {"decision": decision, "raw_output": str(decision)}
        
        job_results[job_id] = {
            "status": "completed",
            "ticker": request.ticker,
            "date": request.date,
            "decision": decision,
            "completed_at": datetime.now().isoformat()
        }
    except Exception as e:
        job_results[job_id] = {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        }

@app.post("/analyze/async")
async def analyze_stock_async(request: TradingRequest, background_tasks: BackgroundTasks):
    import uuid
    job_id = str(uuid.uuid4())
    
    job_results[job_id] = {
        "status": "processing",
        "ticker": request.ticker,
        "date": request.date,
        "started_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(run_analysis_background, job_id, request)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": f"Analysis started for {request.ticker}",
        "check_status_url": f"/jobs/{job_id}"
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_results[job_id]

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Job not found")
    del job_results[job_id]
    return {"message": f"Job {job_id} deleted"}

@app.post("/clear-memory")
async def clear_memory():
    """Clear TradingAgents memory collections to avoid conflicts in batch processing"""
    try:
        import shutil
        import os
        import chromadb
        
        cleared = []
        
        # Method 1: Try to delete ChromaDB collections directly
        try:
            # Find where ChromaDB stores data (check common locations)
            chroma_paths = [
                '.chroma',
                'chroma_db',
                '.chromadb',
                os.path.join(os.getcwd(), '.chroma'),
            ]
            
            for path in chroma_paths:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    cleared.append(path)
                    print(f"✓ Deleted ChromaDB directory: {path}")
        except Exception as e:
            print(f"Error deleting ChromaDB directories: {e}")
        
        # Method 2: Try to delete collections through ChromaDB client
        try:
            client = chromadb.Client()
            collections = client.list_collections()
            for collection in collections:
                try:
                    client.delete_collection(collection.name)
                    cleared.append(f"collection:{collection.name}")
                    print(f"✓ Deleted collection: {collection.name}")
                except:
                    pass
        except Exception as e:
            print(f"Note: Could not access ChromaDB client: {e}")
        
        if cleared:
            return {"message": "Memory cleared", "cleared": cleared}
        else:
            return {"message": "No memory found to clear"}
            
    except Exception as e:
        print(f"Error clearing memory: {e}")
        return {"message": "Memory clear attempted", "error": str(e)}
    
def clear_chromadb_collections():
    """Clear ChromaDB collections and reset the client"""
    try:
        import chromadb
        from chromadb.config import Settings
        import shutil
        import os
        
        # Step 1: Delete physical ChromaDB directories
        chroma_dirs = ['.chroma', 'chroma', '.chromadb', 'chroma_db']
        for dir_name in chroma_dirs:
            if os.path.exists(dir_name):
                try:
                    shutil.rmtree(dir_name)
                    print(f"✓ Deleted directory: {dir_name}")
                except Exception as e:
                    print(f"Could not delete {dir_name}: {e}")
        
        # Step 2: Force reset ChromaDB's internal state
        try:
            # Clear the singleton instance cache
            if hasattr(chromadb, '_client'):
                delattr(chromadb, '_client')
            if hasattr(chromadb.api, '_client'):
                delattr(chromadb.api, '_client')
        except:
            pass
        
        # Step 3: Try to delete all collections from any existing clients
        try:
            import gc
            gc.collect()  # Force garbage collection
        except:
            pass
            
        print("✓ ChromaDB reset complete")
        
    except Exception as e:
        print(f"Error in clear_chromadb_collections: {e}")

@app.get("/tickers")
async def get_tickers():
    """Get saved tickers list"""
    tickers = load_saved_tickers()
    return {"tickers": tickers}

@app.post("/tickers")
async def update_tickers(request: Request):
    """Update saved tickers list"""
    data = await request.json()
    tickers_list = data if isinstance(data, list) else data.get('tickers', [])
    success = save_tickers_to_file(tickers_list)
    if success:
        return {"message": "Tickers saved successfully", "tickers": tickers_list}
    else:
        raise HTTPException(status_code=500, detail="Failed to save tickers")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)