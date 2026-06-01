import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: number | string;
  accent?: "buy" | "sell" | "hold";
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div
          className={cn(
            "text-2xl font-semibold tabular",
            accent === "buy" && "text-buy",
            accent === "sell" && "text-sell",
            accent === "hold" && "text-hold",
          )}
        >
          {value}
        </div>
        <div className="mt-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </div>
      </CardContent>
    </Card>
  );
}
