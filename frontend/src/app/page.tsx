"use client";

import * as React from "react";
import { toast } from "sonner";

import { TickerManager } from "@/components/ticker-manager";
import {
  AnalysisForm,
  type AnalysisConfig,
} from "@/components/analysis-form";
import { AnalysisResult } from "@/components/analysis-result";
import {
  BatchProgress,
  type BatchItem,
} from "@/components/batch-progress";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2 } from "lucide-react";
import { analyze, clearMemory, getTickers } from "@/lib/api";
import type { TradingResponse } from "@/lib/types";

export default function DashboardPage() {
  const [tickers, setTickers] = React.useState<string[]>([]);
  const [loadingTickers, setLoadingTickers] = React.useState(true);
  const [mode, setMode] = React.useState<"single" | "batch">("single");
  const [running, setRunning] = React.useState(false);
  const [results, setResults] = React.useState<TradingResponse[]>([]);
  const [batch, setBatch] = React.useState<BatchItem[] | null>(null);

  React.useEffect(() => {
    getTickers()
      .then((data) => setTickers(data.tickers ?? []))
      .catch(() => setTickers([]))
      .finally(() => setLoadingTickers(false));
  }, []);

  async function onRunSingle(ticker: string, config: AnalysisConfig) {
    setRunning(true);
    setResults([]);
    setBatch(null);
    try {
      const res = await analyze({ ticker, ...config });
      setResults([res]);
      toast.success(`${ticker} analysis complete`);
    } catch (err) {
      toast.error(`${ticker} analysis failed`, {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setRunning(false);
    }
  }

  async function onRunBatch(config: AnalysisConfig) {
    setRunning(true);
    setResults([]);
    const queue: BatchItem[] = tickers.map((ticker) => ({
      ticker,
      status: "pending",
    }));
    setBatch(queue);

    const collected: TradingResponse[] = [];

    for (let i = 0; i < tickers.length; i++) {
      const ticker = tickers[i];
      setBatch((prev) =>
        prev!.map((it, idx) =>
          idx === i ? { ...it, status: "running" } : it,
        ),
      );
      try {
        const res = await analyze({ ticker, ...config });
        collected.push(res);
        setBatch((prev) =>
          prev!.map((it, idx) => (idx === i ? { ...it, status: "done" } : it)),
        );
      } catch (err) {
        setBatch((prev) =>
          prev!.map((it, idx) =>
            idx === i
              ? {
                  ...it,
                  status: "error",
                  error: err instanceof Error ? err.message : "Failed",
                }
              : it,
          ),
        );
      }
      await clearMemory();
      if (i < tickers.length - 1) {
        await new Promise((r) => setTimeout(r, 1500));
      }
    }

    setResults(collected);
    setRunning(false);
    toast.success(`Batch complete · ${collected.length}/${tickers.length}`);
  }

  return (
    <div className="space-y-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-sm text-muted-foreground">
          Run multi-agent trading analysis across your watchlist.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <TickerManager
          tickers={tickers}
          setTickers={setTickers}
          loading={loadingTickers}
        />
        <AnalysisForm
          tickers={tickers}
          mode={mode}
          onModeChange={setMode}
          running={running}
          onRunSingle={onRunSingle}
          onRunBatch={onRunBatch}
        />
      </div>

      {running && !batch && (
        <Card>
          <CardContent className="flex items-center gap-3 py-10">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <div>
              <p className="text-sm font-medium">Running analysis…</p>
              <p className="text-sm text-muted-foreground">
                Agents are debating market conditions. This can take a few
                minutes.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {batch && <BatchProgress items={batch} />}

      {results.length > 0 && (
        <div className="space-y-6">
          {results.map((res, i) => (
            <AnalysisResult key={`${res.ticker}-${i}`} result={res} />
          ))}
        </div>
      )}
    </div>
  );
}
