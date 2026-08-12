import { useEffect, useState } from "react";
import { SidebarProvider, SidebarTrigger } from "./components/ui/sidebar";
import { ConversationSidebar } from "./components/ConversationSidebar";
import { ChatInput } from "./components/ChatInput";
import { ScrollArea } from "./components/ui/scroll-area";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { KnowledgeBaseStats } from "./components/KnowledgeBaseStats";
import { LoginPage } from "./components/LoginPage";
import { AdminDashboard } from "./components/AdminDashboard";
import { Message, ConversationData } from "./lib/mockData";
import { Sparkles, Search, Brain, LogOut } from "lucide-react";
import { Card } from "./components/ui/card";
import { ChatMessage, } from "./components/ChatMessage";
import { sendMessageToRAG, fetchChatHistory } from "./lib/api";
import sabancıLogo from "./assets/sabanci_logo.png";

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState<{ email: string; name: string; isAdmin: boolean } | null>(null);
  const [conversations, setConversations] = useState<ConversationData[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string>("");
  const [isTyping, setIsTyping] = useState(false);


  useEffect(() => {
    const loadHistory = async () => {
      // Only fetch if the user is logged in
      if (isLoggedIn) {
        try {
          // Fetch data from the backend
          const history = await fetchChatHistory();
          console.log("[History] Raw response from backend:", history);

          // Normalize backend response:
          // 1. Convert numeric ids to strings (backend: number, frontend state: string)
          // 2. Convert ISO timestamps to human-readable labels (Istanbul timezone)
          const TZ = "Europe/Istanbul";

          const formatMsgTime = (isoString: string): string => {
            const date = new Date(isoString);
            if (isNaN(date.getTime())) return isoString;
            return date.toLocaleTimeString("tr-TR", {
              hour: "2-digit",
              minute: "2-digit",
              timeZone: TZ,
            });
          };

          const normalize = (isoString: string): string => {
            const date = new Date(isoString);
            if (isNaN(date.getTime())) return isoString;
            const now = new Date();
            const diffDays = Math.floor(
              (now.setHours(0, 0, 0, 0) - new Date(date).setHours(0, 0, 0, 0)) /
              (1000 * 60 * 60 * 24)
            );
            const time = date.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit", timeZone: TZ });
            if (diffDays === 0) return `Bugün ${time}`;
            if (diffDays === 1) return "Dün";
            if (diffDays < 7) return `${diffDays} gün önce`;
            return date.toLocaleDateString("tr-TR", { timeZone: TZ });
          };

          const mapped: ConversationData[] = (history as any[]).map((conv) => ({
            ...conv,
            id: String(conv.id),               // number → string
            timestamp: normalize(conv.timestamp),
            messages: (conv.messages ?? []).map((msg: any) => ({
              ...msg,
              id: String(msg.id),              // number → string
              timestamp: formatMsgTime(msg.timestamp),
              sources: msg.sources || [],      // Include sources from backend
              confidence: msg.confidence,      // Include confidence score
            })),
          }));

          console.log("[History] Mapped conversations:", mapped);
          setConversations(mapped);

          // Only select a conversation if history is non-empty
          if (mapped.length > 0) {
            setCurrentConversationId(mapped[0].id);
          }
        } catch (error) {
          console.error("[History] Error loading chat history:", error);
        }
      }
    };

    loadHistory();
  }, [isLoggedIn]); // Dependency: Re-run when 'isLoggedIn' changes

  const handleLogin = (email: string, name: string, isAdmin: boolean) => {

    setUser({ email, name, isAdmin });
    setIsLoggedIn(true);

  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUser(null);
  };

  if (!isLoggedIn) {
    return <LoginPage onLogin={handleLogin} />;
  }

  // Show admin dashboard for admin users
  if (user?.isAdmin) {
    return <AdminDashboard onLogout={handleLogout} />;
  }

  const currentConversation = conversations.find((c) => c.id === currentConversationId);

  // --- SEND MESSAGE FUNCTION ---
  const handleSendMessage = async (content: string) => {
    // Use a local variable so React state async updates don't cause stale reads
    let convId = currentConversationId;

    // Auto-create a new conversation if none is active
    if (!convId) {
      convId = `new-${Date.now()}`;
      const newConv: ConversationData = {
        id: convId,
        title: content.slice(0, 40) || "New Conversation",
        timestamp: "Just now",
        preview: content.slice(0, 50) + (content.length > 50 ? "..." : ""),
        messages: [],
      };
      setConversations((prev) => [newConv, ...prev]);
      setCurrentConversationId(convId);
    }

    // 1. Immediately display the user's message in the UI
    const newUserMessage: Message = {
      id: `${convId}-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === convId
          ? {
            ...conv,
            messages: [...conv.messages, newUserMessage],
            preview: content.slice(0, 50) + (content.length > 50 ? "..." : ""),
            timestamp: "Just now",
          }
          : conv
      )
    );

    // 2. Show typing indicator
    setIsTyping(true);

    try {
      // 3. Derive numeric conversation ID for the backend (omit for new conversations)
      const numericConvId = convId.startsWith("new-")
        ? undefined
        : parseInt(convId, 10) || undefined;

      // 4. Send request to backend
      const data = await sendMessageToRAG(content, numericConvId);

      // 5. Build the assistant message
      const aiResponse: Message = {
        id: `${convId}-${Date.now()}-ai`,
        role: "assistant",
        content: data.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        sources: data.sources?.map((src) => ({
          chunk_id: src.chunk_id,
          title: src.title,
          excerpt: src.excerpt,
          score: src.score,
          url: src.url,
        })) || [],
        confidence: data.confidence ?? 0.95,
      };

      // 6. Sync the backend-assigned conversation ID and append the AI reply
      const backendConvId = data.conversation_id?.toString() || convId;

      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === convId
            ? { ...conv, id: backendConvId, messages: [...conv.messages, aiResponse] }
            : conv
        )
      );

      if (backendConvId !== convId) {
        setCurrentConversationId(backendConvId);
      }
    } catch (error) {
      console.error("Failed to send message:", error);
    } finally {
      setIsTyping(false);
    }
  };
  // --- END OF FUNCTION ---

  const handleNewConversation = () => {
    const newConv: ConversationData = {
      id: `new-${Date.now()}`,
      title: "New Conversation",
      timestamp: "Just now",
      preview: "Start a new conversation...",
      messages: [],
    };

    setConversations((prev) => [newConv, ...prev]);
    setCurrentConversationId(newConv.id);
  };

  const handleSelectConversation = (id: string) => {
    setCurrentConversationId(id);
  };

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full bg-background">
        <ConversationSidebar
          conversations={conversations.map((c) => ({
            id: c.id,
            title: c.title,
            timestamp: c.timestamp,
            preview: c.preview,
          }))}
          currentConversationId={currentConversationId}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          userName={user?.name || "Student"}
        />

        <div className="flex-1 flex flex-col min-w-0">
          {/* Header */}
          <header className="border-b bg-card px-6 py-3 flex items-center gap-4 shrink-0">
            <SidebarTrigger />
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <img
                src={sabancıLogo}
                alt="Sabancı Universitesi"
                className="h-8 w-auto shrink-0"
              />
              <div className="flex-1 min-w-0">
                <h1 className="flex items-center gap-2 flex-wrap">
                  <span>MACKIS</span>
                  <Badge variant="secondary" className="gap-1 bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                    <Brain className="h-3 w-3" />
                    RAG-Powered
                  </Badge>
                </h1>
                <p className="text-xs text-muted-foreground">
                  Intelligent university assistant with document-sourced answers
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleLogout}
              className="gap-2 shrink-0"
            >
              <LogOut className="h-4 w-4" />
              <span className="hidden sm:inline">Logout</span>
            </Button>
          </header>

          {/* Main Content Area */}
          <div className="flex-1 flex min-h-0">
            {/* Chat Area */}
            <div className="flex-1 flex flex-col min-w-0">
              <ScrollArea className="flex-1">
                {currentConversation && currentConversation.messages.length > 0 ? (
                  <div className="divide-y">
                    {currentConversation.messages.map((message) => (
                      <ChatMessage
                        key={message.id}
                        role={message.role}
                        content={message.content}
                        timestamp={message.timestamp}
                        sources={message.sources || []}
                        confidence={message.confidence || 0}
                      />
                    ))}
                    {isTyping && (
                      <div className="flex gap-4 p-6 bg-muted/30">
                        <div className="h-9 w-9 rounded-full bg-blue-600 flex items-center justify-center shrink-0">
                          <Sparkles className="h-4 w-4 text-white animate-pulse" />
                        </div>
                        <div className="flex-1 space-y-2">
                          <span className="text-sm">MACKIS</span>
                          <div className="flex items-center gap-2">
                            <div className="flex gap-1">
                              <div className="h-2 w-2 rounded-full bg-blue-600/40 animate-bounce [animation-delay:-0.3s]"></div>
                              <div className="h-2 w-2 rounded-full bg-blue-600/40 animate-bounce [animation-delay:-0.15s]"></div>
                              <div className="h-2 w-2 rounded-full bg-blue-600/40 animate-bounce"></div>
                            </div>
                            <span className="text-xs text-muted-foreground">Searching university knowledge base...</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full p-8">
                    <div className="text-center max-w-2xl space-y-6">
                      <img
                        src={sabancıLogo}
                        alt="Sabancı Universitesi"
                        className="h-20 w-auto mx-auto"
                      />
                      <div>
                        <h2 className="mb-2">MACKIS'e Hoş Geldiniz</h2>
                        <p className="text-muted-foreground">
                          Sabancı Üniversitesi'nin akıllı bilgi asistanı. RAG teknolojisiyle desteklenen MACKIS,
                          17,788+ üniversite dokümanına erişerek kaynaklı, doğru yanıtlar sunar.
                        </p>
                      </div>

                      <Card className="p-4 bg-blue-50/50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800">
                        <div className="flex items-start gap-3 text-left">
                          <Search className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 shrink-0" />
                          <div>
                            <h3 className="text-sm mb-1 text-blue-900 dark:text-blue-100">Nasıl Çalışır?</h3>
                            <p className="text-xs text-blue-700 dark:text-blue-300">
                              Sorunuzu sorduğunuzda MACKIS, resmi üniversite yönetmelikleri, el kitapları ve veritabanlarında
                              arama yaparak doğru yanıtı kaynak göstererek sunar.
                            </p>
                          </div>
                        </div>
                      </Card>

                      <div className="grid gap-2 text-left">
                        <button
                          onClick={() => handleSendMessage("Erasmus değişim programı için minimum GNO kaç olmalı?")}
                          className="p-3 rounded-lg border hover:bg-accent transition-colors text-sm text-left group"
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span>✈️</span>
                            <span className="group-hover:text-primary transition-colors">Erasmus değişim programı için minimum GNO kaç olmalı?</span>
                          </div>
                          <p className="text-xs text-muted-foreground pl-6">Uluslararası değişim programı şartları</p>
                        </button>
                        <button
                          onClick={() => handleSendMessage("Kütüphaneden kaç kitap ödünç alabilirim ve iade süresi ne kadar?")}
                          className="p-3 rounded-lg border hover:bg-accent transition-colors text-sm text-left group"
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span>📚</span>
                            <span className="group-hover:text-primary transition-colors">Kütüphaneden kaç kitap ödünç alabilirim ve iade süresi ne kadar?</span>
                          </div>
                          <p className="text-xs text-muted-foreground pl-6">Kütüphane hizmetleri ve kuralları</p>
                        </button>
                        <button
                          onClick={() => handleSendMessage("Uyarı cezası hangi durumlarda verilir?")}
                          className="p-3 rounded-lg border hover:bg-accent transition-colors text-sm text-left group"
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span>⚖️</span>
                            <span className="group-hover:text-primary transition-colors">Uyarı cezası hangi durumlarda verilir?</span>
                          </div>
                          <p className="text-xs text-muted-foreground pl-6">Öğrenci disiplin yönetmeliği</p>
                        </button>
                        <button
                          onClick={() => handleSendMessage("Burs başvurusu için son başvuru tarihi ne zaman?")}
                          className="p-3 rounded-lg border hover:bg-accent transition-colors text-sm text-left group"
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span>🎓</span>
                            <span className="group-hover:text-primary transition-colors">Burs başvurusu için son başvuru tarihi ne zaman?</span>
                          </div>
                          <p className="text-xs text-muted-foreground pl-6">Burs ve mali yardım bilgileri</p>
                        </button>
                        <button
                          onClick={() => handleSendMessage("Çift anadal programına nasıl başvurabilirim?")}
                          className="p-3 rounded-lg border hover:bg-accent transition-colors text-sm text-left group"
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span>📝</span>
                            <span className="group-hover:text-primary transition-colors">Çift anadal programına nasıl başvurabilirim?</span>
                          </div>
                          <p className="text-xs text-muted-foreground pl-6">Akademik program seçenekleri</p>
                        </button>
                        <button
                          onClick={() => handleSendMessage("Mezuniyet için gereken minimum kredi sayısı nedir?")}
                          className="p-3 rounded-lg border hover:bg-accent transition-colors text-sm text-left group"
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span>🏛️</span>
                            <span className="group-hover:text-primary transition-colors">Mezuniyet için gereken minimum kredi sayısı nedir?</span>
                          </div>
                          <p className="text-xs text-muted-foreground pl-6">Lisans mezuniyet gereksinimleri</p>
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </ScrollArea>

              {/* Input Area */}
              <ChatInput
                onSend={handleSendMessage}
                disabled={isTyping}
              />
            </div>

            {/* Right Sidebar - Resources & Actions */}
            <div className="w-80 border-l bg-card/50 p-4 space-y-4 overflow-y-auto shrink-0 hidden xl:block">
              <KnowledgeBaseStats />


            </div>
          </div>
        </div>
      </div>
    </SidebarProvider>
  );
}
