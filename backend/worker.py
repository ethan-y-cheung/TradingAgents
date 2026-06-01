import sys
import json
import os

# FIX: Force UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-your-key-here"  # REPLACE
if not os.getenv("ALPHA_VANTAGE_API_KEY"):
    os.environ["ALPHA_VANTAGE_API_KEY"] = "your-key-here"  # REPLACE

# TEST MODE: Set to True to return fake results without calling APIs
TEST_MODE = False

def run_analysis(config_file):
    """Run a single analysis and return the result"""
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        ticker = config_data['ticker']
        date = config_data['date']
        
        # TEST MODE: Return fake result immediately
        if TEST_MODE:
            decision = {
                "decision": "HOLD",
                "test_mode": True,
                "ticker": ticker,
                "date": date,
                "message": "This is a test result - no API calls made"
            }
            print(json.dumps(decision), flush=True)
            return
        
        # Import here to avoid issues if TEST_MODE is True
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG
        
        # Set up TradingAgents config
        config = DEFAULT_CONFIG.copy()
        config["deep_think_llm"] = config_data['deep_think_llm']
        config["quick_think_llm"] = config_data['quick_think_llm']
        config["max_debate_rounds"] = config_data['max_debate_rounds']
        config["online_tools"] = config_data['online_tools']
        
        # Run analysis (suppress all output except final JSON)
        import io
        import contextlib
        
        # Capture all output during analysis
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ta = TradingAgentsGraph(debug=False, config=config)
            _, decision = ta.propagate(ticker, date)
        
        # Convert to dict if needed
        if not isinstance(decision, dict):
            decision = {"decision": decision, "raw_output": str(decision)}
        
        # Output ONLY the JSON result to stdout (with flush to ensure it's written)
        print(json.dumps(decision), flush=True)
        
    except Exception as e:
        # Print error to stderr (NOT stdout, to keep JSON clean)
        import traceback
        print(f"Error in worker: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: trading_worker.py <config_file>", file=sys.stderr)
        sys.exit(1)
    
    config_file = sys.argv[1]
    run_analysis(config_file)