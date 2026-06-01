import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Markdown } from "@/components/markdown";
import { cn } from "@/lib/utils";

export function ReportSection({
  title,
  content,
  accent,
}: {
  title: string;
  content?: string;
  accent?: "buy" | "sell" | "neutral";
}) {
  return (
    <Card
      className={cn(
        accent === "buy" && "border-l-2 border-l-buy",
        accent === "sell" && "border-l-2 border-l-sell",
        accent === "neutral" && "border-l-2 border-l-hold",
      )}
    >
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {content && content.trim() ? (
          <Markdown content={content} />
        ) : (
          <p className="text-sm text-muted-foreground">No data available.</p>
        )}
      </CardContent>
    </Card>
  );
}
