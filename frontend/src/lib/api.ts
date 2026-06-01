import type {
  AnalysisFile,
  ResultsResponse,
  TickersResponse,
  TradingRequest,
  TradingResponse,
} from "./types";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail);
  }

  return res.json() as Promise<T>;
}

export function getTickers(): Promise<TickersResponse> {
  return request<TickersResponse>("/tickers");
}

export function saveTickers(tickers: string[]): Promise<TickersResponse> {
  return request<TickersResponse>("/tickers", {
    method: "POST",
    body: JSON.stringify({ tickers }),
  });
}

export function analyze(req: TradingRequest): Promise<TradingResponse> {
  return request<TradingResponse>("/analyze", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function clearMemory(): Promise<unknown> {
  return request<unknown>("/clear-memory", { method: "POST" }).catch(() => null);
}

export function getResults(): Promise<ResultsResponse> {
  return request<ResultsResponse>("/results");
}

export function getResult(ticker: string, date: string): Promise<AnalysisFile> {
  return request<AnalysisFile>(
    `/results/${encodeURIComponent(ticker)}/${encodeURIComponent(date)}`,
  );
}
