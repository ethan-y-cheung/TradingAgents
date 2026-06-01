import { ArrowDownRight, ArrowUpRight, Minus, CircleHelp } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Decision } from "@/lib/types";

const CONFIG: Record<
  Decision,
  { variant: "buy" | "sell" | "hold" | "muted"; Icon: typeof Minus }
> = {
  BUY: { variant: "buy", Icon: ArrowUpRight },
  SELL: { variant: "sell", Icon: ArrowDownRight },
  HOLD: { variant: "hold", Icon: Minus },
  UNKNOWN: { variant: "muted", Icon: CircleHelp },
};

export function DecisionBadge({
  decision,
  className,
}: {
  decision: Decision;
  className?: string;
}) {
  const { variant, Icon } = CONFIG[decision] ?? CONFIG.UNKNOWN;
  return (
    <Badge variant={variant} className={cn("gap-1", className)}>
      <Icon className="h-3 w-3" aria-hidden />
      {decision}
    </Badge>
  );
}
