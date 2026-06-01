"use client";

import * as React from "react";
import { Copy, Check } from "lucide-react";
import { toast } from "sonner";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { DecisionBadge } from "@/components/decision-badge";
import { extractDecision } from "@/lib/decision";
import type { TradingResponse } from "@/lib/types";

function formatKey(key: string) {
  return key
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export function AnalysisResult({ result }: { result: TradingResponse }) {
  const [copied, setCopied] = React.useState(false);
  const decisionObj =
    typeof result.decision === "object" && result.decision !== null
      ? (result.decision as Record<string, unknown>)
      : null;
  const json = JSON.stringify(result.decision, null, 2);
  const signal = extractDecision(json);

  function copy() {
    void navigator.clipboard.writeText(json);
    setCopied(true);
    toast.success("Copied to clipboard");
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-3">
          <span className="font-mono text-lg tabular">{result.ticker}</span>
          <span className="text-sm font-normal text-muted-foreground tabular">
            {result.date}
          </span>
        </CardTitle>
        <DecisionBadge decision={signal} />
      </CardHeader>
      <CardContent className="space-y-4">
        {decisionObj ? (
          <div className="space-y-4">
            {Object.entries(decisionObj).map(([key, value]) => (
              <div key={key} className="space-y-1">
                <h4 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  {formatKey(key)}
                </h4>
                <p className="whitespace-pre-wrap text-sm leading-relaxed">
                  {typeof value === "object"
                    ? JSON.stringify(value, null, 2)
                    : String(value)}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="whitespace-pre-wrap text-sm leading-relaxed">
            {String(result.decision)}
          </p>
        )}

        <Separator />

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Raw output
            </h4>
            <Button variant="ghost" size="sm" onClick={copy}>
              {copied ? (
                <Check className="h-3.5 w-3.5" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
              Copy
            </Button>
          </div>
          <pre className="max-h-80 overflow-auto rounded-md bg-muted p-4 font-mono text-xs leading-relaxed">
            {json}
          </pre>
        </div>
      </CardContent>
    </Card>
  );
}
