import { Card } from "./ui/card";
import { FileText, ExternalLink } from "lucide-react";

// Define the source reference type matching backend schema
export interface SourceReference {
  chunk_id: number;
  title: string;
  excerpt: string;
  score?: number;
  url?: string;
}

interface SourceCardProps {
  source: SourceReference;
  index?: number;
}

export function SourceCard({ source, index }: SourceCardProps) {
  const handleClick = () => {
    if (source.url) {
      window.open(source.url, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <Card
      className={`p-3 hover:bg-muted/50 transition-colors border shadow-sm group bg-card ${source.url ? 'cursor-pointer' : ''}`}
      onClick={source.url ? handleClick : undefined}
    >
      <div className="flex items-start gap-3">
        {/* Icon Section */}
        <div className="mt-1 bg-blue-100 dark:bg-blue-900/30 p-1.5 rounded-md text-blue-600 dark:text-blue-400">
          <FileText className="h-4 w-4" />
        </div>

        {/* Content Section */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start gap-2">
            <h4 className="text-sm font-medium text-foreground line-clamp-1 group-hover:text-primary transition-colors" title={source.title}>
              {index !== undefined ? `[${index + 1}] ` : ""}{source.title}
            </h4>
          </div>

          {/* Metadata Row — show link hint only when URL is available */}
          {source.url && (
            <div className="flex items-center gap-1 mt-1.5 text-[10px] text-muted-foreground">
              <ExternalLink className="h-3 w-3" />
              <span>Kaynağa git</span>
            </div>
          )}

          {/* Excerpt preview */}
          {source.excerpt && (
            <p className="mt-2 text-xs text-muted-foreground line-clamp-2 italic border-l-2 pl-2">
              "{source.excerpt.substring(0, 150)}..."
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}