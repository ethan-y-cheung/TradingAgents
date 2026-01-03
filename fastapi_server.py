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
        
        return TradingResponse(
            ticker=request.ticker,
            date=request.date,
            decision=decision,
            status="completed"
        )
    except Exception as e:
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)