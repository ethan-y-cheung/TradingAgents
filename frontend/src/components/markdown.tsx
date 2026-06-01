import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

export function Markdown({
  content,
  className,
}: {
  content: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "prose-trading max-w-none text-sm leading-relaxed text-foreground",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ ...props }) => (
            <h1 className="mb-3 mt-5 text-lg font-semibold" {...props} />
          ),
          h2: ({ ...props }) => (
            <h2 className="mb-2 mt-4 text-base font-semibold" {...props} />
          ),
          h3: ({ ...props }) => (
            <h3 className="mb-2 mt-3 text-sm font-semibold" {...props} />
          ),
          p: ({ ...props }) => <p className="mb-3 last:mb-0" {...props} />,
          ul: ({ ...props }) => (
            <ul className="mb-3 list-disc space-y-1 pl-5" {...props} />
          ),
          ol: ({ ...props }) => (
            <ol className="mb-3 list-decimal space-y-1 pl-5" {...props} />
          ),
          strong: ({ ...props }) => (
            <strong className="font-semibold text-foreground" {...props} />
          ),
          a: ({ ...props }) => (
            <a
              className="font-medium text-primary underline underline-offset-4"
              target="_blank"
              rel="noreferrer"
              {...props}
            />
          ),
          code: ({ ...props }) => (
            <code
              className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs"
              {...props}
            />
          ),
          table: ({ ...props }) => (
            <div className="my-4 w-full overflow-x-auto">
              <table
                className="w-full border-collapse text-left text-xs tabular"
                {...props}
              />
            </div>
          ),
          th: ({ ...props }) => (
            <th
              className="border-b px-3 py-2 font-medium text-muted-foreground"
              {...props}
            />
          ),
          td: ({ ...props }) => (
            <td className="border-b px-3 py-2 align-top" {...props} />
          ),
          blockquote: ({ ...props }) => (
            <blockquote
              className="my-3 border-l-2 pl-4 italic text-muted-foreground"
              {...props}
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
