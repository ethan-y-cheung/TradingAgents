"use client";

import { Loader2, CheckCircle2, XCircle, Circle } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export type BatchItemStatus = "pending" | "running" | "done" | "error";

export interface BatchItem {
  ticker: string;
  status: BatchItemStatus;
  error?: string;
}

const ICONS: Record<BatchItemStatus, React.ReactNode> = {
  pending: <Circle className="h-4 w-4 text-muted-foreground" />,
  running: <Loader2 className="h-4 w-4 animate-spin text-primary" />,
  done: <CheckCircle2 className="h-4 w-4 text-buy" />,
  error: <XCircle className="h-4 w-4 text-sell" />,
};

export function BatchProgress({ items }: { items: BatchItem[] }) {
  const done = items.filter(
    (i) => i.status === "done" || i.status === "error",
  ).length;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          Batch analysis
        </CardTitle>
        <CardDescription className="tabular">
          {done} / {items.length} complete
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2">
          {items.map((item) => (
            <li
              key={item.ticker}
              className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"
            >
              <span className="flex items-center gap-2">
                {ICONS[item.status]}
                <span className="font-mono tabular">{item.ticker}</span>
              </span>
              {item.error ? (
                <span className="max-w-[60%] truncate text-xs text-sell">
                  {item.error}
                </span>
              ) : (
                <span className="text-xs capitalize text-muted-foreground">
                  {item.status}
                </span>
              )}
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
