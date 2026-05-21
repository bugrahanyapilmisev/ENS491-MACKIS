import { useState } from "react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Send, ChevronDown } from "lucide-react";
import React from 'react';
import { ChatModel, MODEL_OPTIONS } from "../lib/api";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  selectedModel: ChatModel;
  onModelChange: (model: ChatModel) => void;
}

export function ChatInput({ onSend, disabled, selectedModel, onModelChange }: ChatInputProps) {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const currentOption = MODEL_OPTIONS.find((m) => m.value === selectedModel);

  return (
    <form onSubmit={handleSubmit} className="border-t bg-background p-4 space-y-2">
      {/* Model selector row */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Model:</span>
        <div className="relative">
          <select
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value as ChatModel)}
            disabled={disabled}
            className="appearance-none text-xs font-medium bg-muted/60 border border-border rounded-md pl-3 pr-7 py-1.5 cursor-pointer hover:bg-muted transition-colors focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {MODEL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
        </div>
        {currentOption?.hint && (
          <span className="text-[10px] text-muted-foreground/70">{currentOption.hint}</span>
        )}
      </div>

      {/* Text input row */}
      <div className="flex gap-2">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about the university..."
          className="min-h-[60px] max-h-[200px] resize-none"
          disabled={disabled}
        />
        <Button
          type="submit"
          size="icon"
          className="h-[60px] w-[60px] shrink-0"
          disabled={disabled || !input.trim()}
        >
          <Send className="h-5 w-5" />
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Press Enter to send, Shift + Enter for new line
      </p>
    </form>
  );
}
