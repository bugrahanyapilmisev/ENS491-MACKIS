import { useEffect, useState } from "react";
import { Card } from "./ui/card";
import { Loader2 } from "lucide-react";

interface KBStats {
  document_count: number;
  topic_count: number;
}

function fmt(n: number): string {
  if (n >= 1_000) return `${n.toLocaleString("tr-TR")}+`;
  return String(n);
}

export function KnowledgeBaseStats() {
  const [stats, setStats] = useState<KBStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();

    fetch("/api/stats", { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<KBStats>;
      })
      .then((data) => {
        setStats(data);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          console.error("[KnowledgeBaseStats] fetch failed:", err);
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, []);

  return (
    <Card className="p-4">
      <h3 className="mb-1 text-sm">Bilgi Tabanı</h3>
      <p className="text-xs text-muted-foreground mb-3">
        Sabancı Üniversitesi resmi kaynakları
      </p>

      {loading ? (
        <div className="flex items-center gap-2 text-muted-foreground py-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-xs">Yükleniyor…</span>
        </div>
      ) : stats ? (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Kaynak doküman</span>
            <span className="text-sm font-medium">{fmt(stats.document_count)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Kapsanan konu</span>
            <span className="text-sm font-medium">{fmt(stats.topic_count)}</span>
          </div>
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">Veriler yüklenemedi.</p>
      )}
    </Card>
  );
}
