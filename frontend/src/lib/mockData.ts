import { SourceReference } from "../components/SourceCard";

// Message interface update
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  sources?: SourceReference[]; // Using the new type
  confidence?: number;
  category?: string;
}

export interface ConversationData {
  id: string;
  title: string;
  timestamp: string;
  preview: string;
  messages: Message[];
  category?: string; // "academic", "campus", "financial" etc.
}

export const categories = [
  { id: "all", label: "Tüm Konular", icon: "📋" },
  { id: "exchange", label: "Değişim Programları", icon: "✈️" },
  { id: "academic", label: "Akademik", icon: "🎓" },
  { id: "library", label: "Kütüphane", icon: "📚" },
  { id: "scholarships", label: "Burslar", icon: "💰" },
  { id: "discipline", label: "Disiplin", icon: "⚖️" },
];