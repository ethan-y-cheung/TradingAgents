export type Decision = "BUY" | "SELL" | "HOLD" | "UNKNOWN";

export interface TickersResponse {
  tickers: string[];
}

export type LlmModel = "gpt-4o-mini" | "gpt-4o" | "gpt-4";

export interface TradingRequest {
  ticker: string;
  date: string;
  deep_think_llm: LlmModel;
  quick_think_llm: LlmModel;
  max_debate_rounds: number;
  online_tools: boolean;
}

export interface TradingResponse {
  ticker: string;
  date: string;
  decision: Record<string, unknown> | string;
  status: string;
}

export interface ResultSummary {
  date: string;
  decision: Decision;
}

export interface ResultsResponse {
  results: Record<string, ResultSummary[]>;
}

export interface InvestmentDebateState {
  bull_history?: string;
  bear_history?: string;
  judge_decision?: string;
}

export interface RiskDebateState {
  risky_history?: string;
  safe_history?: string;
  neutral_history?: string;
  judge_decision?: string;
}

export interface AnalysisState {
  company_of_interest?: string;
  trade_date?: string;
  final_trade_decision?: string;
  market_report?: string;
  sentiment_report?: string;
  news_report?: string;
  fundamentals_report?: string;
  trader_investment_decision?: string;
  investment_plan?: string;
  investment_debate_state?: InvestmentDebateState;
  risk_debate_state?: RiskDebateState;
  [key: string]: unknown;
}

export type AnalysisFile = Record<string, AnalysisState>;
