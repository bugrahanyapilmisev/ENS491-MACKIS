import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Bot, User, CheckCircle2 } from "lucide-react";
import { Badge } from "./ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "./ui/collapsible";

// ✅ Import SourceCard and the Type from the file we just updated
import { SourceCard, SourceReference } from "./SourceCard";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  sources?: SourceReference[]; // Now using the shared type
  confidence?: number;
}

export function ChatMessage({ role, content, timestamp, sources, confidence }: ChatMessageProps) {
  const isAssistant = role === "assistant";
  const [sourcesOpen, setSourcesOpen] = useState(false);

  // Helper to display confidence score
  const getConfidenceBadge = () => {
    if (!confidence) return null;
    
    const percentage = Math.round(confidence * 100);
    let color = "";
    
    if (percentage >= 90) {
      color = "bg-green-500/10 text-green-700 dark:text-green-400 border-green-500/20";
    } else if (percentage >= 75) {
      color = "bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-500/20";
    } else {
      color = "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400 border-yellow-500/20";
    }

    return (
      <Badge variant="outline" className={`text-xs gap-1 ${color}`}>
        <CheckCircle2 className="h-3 w-3" />
        {percentage}% confidence
      </Badge>
    );
  };

  return (
    <div className={`flex gap-4 p-6 transition-colors ${isAssistant ? "bg-muted/30" : "bg-background"}`}>
      <Avatar className="h-9 w-9 shrink-0 shadow-sm">
        <AvatarFallback className={isAssistant ? "bg-blue-600 text-white" : "bg-slate-200 dark:bg-slate-700"}>
          {isAssistant ? <Bot className="h-5 w-5" /> : <User className="h-5 w-5" />}
        </AvatarFallback>
      </Avatar>

      <div className="flex-1 space-y-3 max-w-4xl">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-semibold text-sm text-foreground">
            {isAssistant ? "MACKIS" : "You"}
          </span>
          {timestamp && (
            <span className="text-xs text-muted-foreground">{timestamp}</span>
          )}
          {isAssistant && getConfidenceBadge()}
        </div>

        <div className="text-sm text-foreground/90 leading-relaxed prose dark:prose-invert max-w-none">
          {isAssistant ? (
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({node, ...props}) => <a {...props} className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer" />,
                code: ({node, ...props}) => <code {...props} className="bg-muted px-1 py-0.5 rounded text-xs font-mono" />
              }}
            >
              {content}
            </ReactMarkdown>
          ) : (
            <span className="whitespace-pre-wrap">{content}</span>
          )}
        </div>
        
        {/* Sources Section */}
        {isAssistant && sources && sources.length > 0 && (
          <Collapsible open={sourcesOpen} onOpenChange={setSourcesOpen} className="mt-4">
            <CollapsibleTrigger className="flex items-center gap-2 text-xs font-medium text-muted-foreground hover:text-primary transition-colors group">
              <div className="flex items-center gap-1.5 bg-muted/50 px-2 py-1 rounded-md border border-transparent group-hover:border-border">
                <span>📚 Referenced {sources.length} document{sources.length > 1 ? 's' : ''}</span>
                <span className="text-[10px] opacity-70 transition-transform duration-200" style={{ transform: sourcesOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>
                  ▼
                </span>
              </div>
            </CollapsibleTrigger>
            
            <CollapsibleContent className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3 animate-in slide-in-from-top-2 fade-in duration-200">
              {sources.map((source, index) => (
                <SourceCard key={index} source={source} index={index} />
              ))}
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
}