"use client";

import * as React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Play, Target, Layers } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import type { LlmModel, TradingRequest } from "@/lib/types";

const MODELS: { value: LlmModel; label: string }[] = [
  { value: "gpt-5.4-nano", label: "GPT-5.4 Nano · Cheapest" },
  { value: "gpt-5.4-mini", label: "GPT-5.4 Mini · Cheap" },
  { value: "gpt-5.4", label: "GPT-5.4 · Balanced" },
  { value: "gpt-5.5", label: "GPT-5.5 · Deep" },
  { value: "gpt-5.5-pro", label: "GPT-5.5 Pro · Premium" },
];

const ROUNDS = [
  { value: "1", label: "1 · Fastest" },
  { value: "2", label: "2 · Balanced" },
  { value: "3", label: "3 · Thorough" },
  { value: "5", label: "5 · Maximum" },
];

const schema = z.object({
  ticker: z.string(),
  date: z.string().min(1, "Select an analysis date"),
  deep_think_llm: z.enum([
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.5-pro",
  ]),
  quick_think_llm: z.enum([
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.5-pro",
  ]),
  max_debate_rounds: z.coerce.number().int().min(1),
  online_tools: z.boolean(),
});

type FormValues = z.infer<typeof schema>;

export type AnalysisConfig = Omit<TradingRequest, "ticker">;

export function AnalysisForm({
  tickers,
  mode,
  onModeChange,
  running,
  onRunSingle,
  onRunBatch,
}: {
  tickers: string[];
  mode: "single" | "batch";
  onModeChange: (mode: "single" | "batch") => void;
  running: boolean;
  onRunSingle: (ticker: string, config: AnalysisConfig) => void;
  onRunBatch: (config: AnalysisConfig) => void;
}) {
  const today = new Date().toISOString().slice(0, 10);

  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      ticker: "",
      date: today,
      deep_think_llm: "gpt-5.5",
      quick_think_llm: "gpt-5.4-mini",
      max_debate_rounds: 1,
      online_tools: false,
    },
  });

  const ticker = watch("ticker");

  function submit(values: FormValues) {
    const config: AnalysisConfig = {
      date: values.date,
      deep_think_llm: values.deep_think_llm,
      quick_think_llm: values.quick_think_llm,
      max_debate_rounds: values.max_debate_rounds,
      online_tools: values.online_tools,
    };
    if (mode === "single") {
      onRunSingle(values.ticker, config);
    } else {
      onRunBatch(config);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Run Analysis</CardTitle>
        <CardDescription>
          Configure the multi-agent debate and execute against your symbols.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mb-6 grid grid-cols-2 gap-2">
          <ModeButton
            active={mode === "single"}
            onClick={() => onModeChange("single")}
            icon={<Target className="h-4 w-4" />}
            label="Single ticker"
          />
          <ModeButton
            active={mode === "batch"}
            onClick={() => onModeChange("batch")}
            icon={<Layers className="h-4 w-4" />}
            label="Entire watchlist"
          />
        </div>

        <form onSubmit={handleSubmit(submit)} className="space-y-5">
          <div className="grid gap-5 sm:grid-cols-2">
            {mode === "single" && (
              <div className="space-y-2">
                <Label htmlFor="ticker">Ticker</Label>
                <Select
                  value={ticker}
                  onValueChange={(v) => setValue("ticker", v)}
                >
                  <SelectTrigger
                    id="ticker"
                    className="font-mono tabular"
                    aria-invalid={!ticker}
                  >
                    <SelectValue placeholder="Select a symbol" />
                  </SelectTrigger>
                  <SelectContent>
                    {tickers.length === 0 ? (
                      <SelectItem value="__none" disabled>
                        Add a symbol first
                      </SelectItem>
                    ) : (
                      tickers.map((t) => (
                        <SelectItem key={t} value={t} className="font-mono">
                          {t}
                        </SelectItem>
                      ))
                    )}
                  </SelectContent>
                </Select>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="date">Analysis date</Label>
              <Input
                id="date"
                type="date"
                className="tabular"
                {...register("date")}
              />
              {errors.date && (
                <p className="text-xs text-destructive">
                  {errors.date.message}
                </p>
              )}
            </div>
          </div>

          <div className="grid gap-5 sm:grid-cols-2">
            <ModelSelect
              id="deep_think_llm"
              label="Deep-think model"
              value={watch("deep_think_llm")}
              onChange={(v) => setValue("deep_think_llm", v)}
            />
            <ModelSelect
              id="quick_think_llm"
              label="Quick-think model"
              value={watch("quick_think_llm")}
              onChange={(v) => setValue("quick_think_llm", v)}
            />
          </div>

          <div className="grid gap-5 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="max_debate_rounds">Debate rounds</Label>
              <Select
                value={String(watch("max_debate_rounds"))}
                onValueChange={(v) => setValue("max_debate_rounds", Number(v))}
              >
                <SelectTrigger id="max_debate_rounds">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ROUNDS.map((r) => (
                    <SelectItem key={r.value} value={r.value}>
                      {r.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-end">
              <label className="flex w-full items-center justify-between rounded-md border px-3 py-2">
                <span className="space-y-0.5">
                  <span className="block text-sm font-medium">
                    Real-time data
                  </span>
                  <span className="block text-xs text-muted-foreground">
                    Use live market tools
                  </span>
                </span>
                <Switch
                  checked={watch("online_tools")}
                  onCheckedChange={(v) => setValue("online_tools", v)}
                  aria-label="Use real-time data"
                />
              </label>
            </div>
          </div>

          <Button
            type="submit"
            className="w-full"
            size="lg"
            disabled={
              running ||
              (mode === "single" && !ticker) ||
              (mode === "batch" && tickers.length === 0)
            }
          >
            <Play className="h-4 w-4" />
            {mode === "single"
              ? "Run analysis"
              : `Analyze ${tickers.length} symbol${tickers.length === 1 ? "" : "s"}`}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function ModeButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "flex items-center justify-center gap-2 rounded-md border px-4 py-2.5 text-sm font-medium transition-colors",
        active
          ? "border-primary bg-primary text-primary-foreground"
          : "hover:bg-accent",
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function ModelSelect({
  id,
  label,
  value,
  onChange,
}: {
  id: string;
  label: string;
  value: LlmModel;
  onChange: (value: LlmModel) => void;
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <Select value={value} onValueChange={(v) => onChange(v as LlmModel)}>
        <SelectTrigger id={id}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {MODELS.map((m) => (
            <SelectItem key={m.value} value={m.value}>
              {m.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
