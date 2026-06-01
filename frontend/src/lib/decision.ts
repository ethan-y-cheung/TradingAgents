import type { Decision } from "./types";

export function extractDecision(text: string | undefined | null): Decision {
  if (!text) return "UNKNOWN";
  const match = text.match(/\*\*(BUY|SELL|HOLD)\*\*/i);
  if (match) return match[1].toUpperCase() as Decision;
  const upper = text.toUpperCase();
  if (upper.includes("SELL")) return "SELL";
  if (upper.includes("BUY")) return "BUY";
  if (upper.includes("HOLD")) return "HOLD";
  return "UNKNOWN";
}
