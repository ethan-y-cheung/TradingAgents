"use client";

import * as React from "react";
import { Plus, X } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { saveTickers } from "@/lib/api";

export function TickerManager({
  tickers,
  setTickers,
  loading,
}: {
  tickers: string[];
  setTickers: (next: string[]) => void;
  loading: boolean;
}) {
  const [value, setValue] = React.useState("");

  async function persist(next: string[]) {
    setTickers(next);
    try {
      await saveTickers(next);
    } catch (err) {
      toast.error("Failed to save tickers", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  function addTicker() {
    const symbol = value.trim().toUpperCase();
    if (!symbol) return;
    if (tickers.includes(symbol)) {
      toast.warning(`${symbol} is already in your watchlist`);
      return;
    }
    void persist([...tickers, symbol]);
    setValue("");
  }

  function removeTicker(symbol: string) {
    void persist(tickers.filter((t) => t !== symbol));
    toast.success(`Removed ${symbol}`);
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Watchlist</CardTitle>
        <CardDescription>
          Symbols available for single and batch analysis.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                addTicker();
              }
            }}
            placeholder="Add a symbol, e.g. NVDA"
            className="font-mono uppercase tabular"
            aria-label="New ticker symbol"
          />
          <Button onClick={addTicker} type="button">
            <Plus className="h-4 w-4" />
            Add
          </Button>
        </div>

        {loading ? (
          <div className="flex flex-wrap gap-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-7 w-20 rounded-md" />
            ))}
          </div>
        ) : tickers.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No symbols yet. Add one above to get started.
          </p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {tickers.map((symbol) => (
              <Badge
                key={symbol}
                variant="secondary"
                className="gap-1.5 py-1 pl-2.5 pr-1 font-mono text-sm tabular"
              >
                {symbol}
                <button
                  type="button"
                  onClick={() => removeTicker(symbol)}
                  className="rounded-sm p-0.5 text-muted-foreground transition-colors hover:bg-destructive hover:text-destructive-foreground"
                  aria-label={`Remove ${symbol}`}
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
