"use client";

import * as React from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, AlertTriangle } from "lucide-react";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { DecisionBadge } from "@/components/decision-badge";
import { ReportSection } from "@/components/report-section";
import { getResult } from "@/lib/api";
import { extractDecision } from "@/lib/decision";
import type { AnalysisState } from "@/lib/types";

const TABS = [
  { value: "summary", label: "Summary" },
  { value: "technical", label: "Technical" },
  { value: "sentiment", label: "Sentiment" },
  { value: "news", label: "News" },
  { value: "fundamentals", label: "Fundamentals" },
  { value: "debate", label: "Debate" },
  { value: "risk", label: "Risk" },
  { value: "raw", label: "Raw" },
];

function AnalysisView() {
  const params = useSearchParams();
  const ticker = params.get("ticker");
  const date = params.get("date");

  const [data, setData] = React.useState<AnalysisState | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!ticker || !date) {
      setError("Missing ticker or date.");
      setLoading(false);
      return;
    }
    getResult(ticker, date)
      .then((file) => {
        const first = Object.values(file)[0];
        if (!first) throw new Error("No analysis data found.");
        setData(first);
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load."),
      )
      .finally(() => setLoading(false));
  }, [ticker, date]);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-9 w-40" />
        <Skeleton className="h-28 w-full rounded-xl" />
        <Skeleton className="h-10 w-full rounded-lg" />
        <Skeleton className="h-64 w-full rounded-xl" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center gap-3 py-16 text-center">
          <AlertTriangle className="h-8 w-8 text-sell" />
          <div>
            <p className="font-medium">Couldn&apos;t load analysis</p>
            <p className="text-sm text-muted-foreground">{error}</p>
          </div>
          <Button asChild variant="outline" size="sm">
            <Link href="/history">
              <ArrowLeft className="h-4 w-4" />
              Back to history
            </Link>
          </Button>
        </CardContent>
      </Card>
    );
  }

  const decision = extractDecision(data.final_trade_decision);
  const debate = data.investment_debate_state ?? {};
  const risk = data.risk_debate_state ?? {};

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <Button asChild variant="ghost" size="sm">
          <Link href="/history">
            <ArrowLeft className="h-4 w-4" />
            History
          </Link>
        </Button>
      </div>

      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle className="font-mono text-2xl tabular">
              {data.company_of_interest ?? ticker}
            </CardTitle>
            <p className="mt-1 text-sm text-muted-foreground tabular">
              {data.trade_date ?? date}
            </p>
          </div>
          <DecisionBadge decision={decision} className="px-3 py-1 text-sm" />
        </CardHeader>
      </Card>

      <Tabs defaultValue="summary">
        <TabsList className="flex h-auto w-full flex-wrap justify-start gap-1">
          {TABS.map((t) => (
            <TabsTrigger key={t.value} value={t.value}>
              {t.label}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="summary" className="space-y-4">
          <ReportSection
            title="Trader Investment Decision"
            content={data.trader_investment_decision}
          />
          <ReportSection
            title="Investment Plan"
            content={data.investment_plan}
          />
          <ReportSection
            title="Final Trade Decision"
            content={data.final_trade_decision}
          />
        </TabsContent>

        <TabsContent value="technical">
          <ReportSection
            title="Technical Analysis"
            content={data.market_report}
          />
        </TabsContent>

        <TabsContent value="sentiment">
          <ReportSection
            title="Sentiment Analysis"
            content={data.sentiment_report}
          />
        </TabsContent>

        <TabsContent value="news">
          <ReportSection
            title="News & Macroeconomics"
            content={data.news_report}
          />
        </TabsContent>

        <TabsContent value="fundamentals">
          <ReportSection
            title="Fundamental Analysis"
            content={data.fundamentals_report}
          />
        </TabsContent>

        <TabsContent value="debate" className="space-y-4">
          <ReportSection
            title="🐂 Bull Analyst"
            content={debate.bull_history}
            accent="buy"
          />
          <ReportSection
            title="🐻 Bear Analyst"
            content={debate.bear_history}
            accent="sell"
          />
          <ReportSection
            title="⚖️ Judge's Decision"
            content={debate.judge_decision}
          />
        </TabsContent>

        <TabsContent value="risk" className="space-y-4">
          <ReportSection
            title="Risky Analyst"
            content={risk.risky_history}
            accent="sell"
          />
          <ReportSection
            title="Safe Analyst"
            content={risk.safe_history}
            accent="buy"
          />
          <ReportSection
            title="Neutral Analyst"
            content={risk.neutral_history}
            accent="neutral"
          />
          <ReportSection
            title="Final Risk Assessment"
            content={risk.judge_decision}
          />
        </TabsContent>

        <TabsContent value="raw">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Raw Data</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="max-h-[600px] overflow-auto rounded-md bg-muted p-4 font-mono text-xs leading-relaxed">
                {JSON.stringify(data, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default function AnalysisPage() {
  return (
    <React.Suspense
      fallback={<Skeleton className="h-64 w-full rounded-xl" />}
    >
      <AnalysisView />
    </React.Suspense>
  );
}
