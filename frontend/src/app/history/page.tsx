"use client";

import * as React from "react";
import Link from "next/link";
import { Search, Inbox, AlertTriangle } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { StatCard } from "@/components/stat-card";
import { DecisionBadge } from "@/components/decision-badge";
import { getResults } from "@/lib/api";
import type { ResultSummary } from "@/lib/types";

type ResultsMap = Record<string, ResultSummary[]>;

export default function HistoryPage() {
  const [results, setResults] = React.useState<ResultsMap>({});
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [query, setQuery] = React.useState("");

  React.useEffect(() => {
    getResults()
      .then((data) => setResults(data.results ?? {}))
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load history"),
      )
      .finally(() => setLoading(false));
  }, []);

  const tickers = React.useMemo(
    () =>
      Object.keys(results)
        .filter((t) => t.toLowerCase().includes(query.toLowerCase()))
        .sort(),
    [results, query],
  );

  const stats = React.useMemo(() => {
    let total = 0;
    let buy = 0;
    let sell = 0;
    let hold = 0;
    for (const list of Object.values(results)) {
      total += list.length;
      for (const a of list) {
        if (a.decision === "BUY") buy++;
        else if (a.decision === "SELL") sell++;
        else if (a.decision === "HOLD") hold++;
      }
    }
    return { tickers: Object.keys(results).length, total, buy, sell, hold };
  }, [results]);

  return (
    <div className="space-y-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">History</h1>
        <p className="text-sm text-muted-foreground">
          Browse past multi-agent analyses by symbol.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        {loading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-[88px] rounded-xl" />
          ))
        ) : (
          <>
            <StatCard label="Symbols" value={stats.tickers} />
            <StatCard label="Analyses" value={stats.total} />
            <StatCard label="Buy" value={stats.buy} accent="buy" />
            <StatCard label="Sell" value={stats.sell} accent="sell" />
            <StatCard label="Hold" value={stats.hold} accent="hold" />
          </>
        )}
      </div>

      <Card>
        <CardContent className="space-y-4 p-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by symbol…"
              className="pl-9 font-mono uppercase tabular"
              aria-label="Filter by symbol"
            />
          </div>

          {loading ? (
            <div className="space-y-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full rounded-md" />
              ))}
            </div>
          ) : error ? (
            <EmptyState
              icon={<AlertTriangle className="h-8 w-8 text-sell" />}
              title="Couldn't load history"
              description={error}
            />
          ) : tickers.length === 0 ? (
            <EmptyState
              icon={<Inbox className="h-8 w-8 text-muted-foreground" />}
              title="No analyses found"
              description="Run an analysis from the dashboard to populate history."
            />
          ) : (
            <Accordion type="multiple" className="w-full">
              {tickers.map((ticker) => (
                <AccordionItem key={ticker} value={ticker}>
                  <AccordionTrigger>
                    <span className="flex items-center gap-3">
                      <span className="font-mono text-base font-semibold tabular">
                        {ticker}
                      </span>
                      <Badge variant="muted" className="tabular">
                        {results[ticker].length}
                      </Badge>
                    </span>
                  </AccordionTrigger>
                  <AccordionContent>
                    <ul className="space-y-2">
                      {results[ticker].map((a) => (
                        <li key={`${ticker}-${a.date}`}>
                          <Link
                            href={`/analysis?ticker=${encodeURIComponent(
                              ticker,
                            )}&date=${encodeURIComponent(a.date)}`}
                            className="flex items-center justify-between rounded-md border px-4 py-3 transition-colors hover:bg-accent"
                          >
                            <span className="font-mono text-sm tabular">
                              {a.date}
                            </span>
                            <DecisionBadge decision={a.decision} />
                          </Link>
                        </li>
                      ))}
                    </ul>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function EmptyState({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="flex flex-col items-center gap-3 py-16 text-center">
      {icon}
      <div>
        <p className="font-medium">{title}</p>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
    </div>
  );
}
