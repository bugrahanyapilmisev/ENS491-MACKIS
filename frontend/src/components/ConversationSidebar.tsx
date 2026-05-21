import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "./ui/sidebar";
import { MessageSquare, Plus, User, LogOut } from "lucide-react";
import { Button } from "./ui/button";
import { Separator } from "./ui/separator";

export interface Conversation {
  id: string;
  title: string;
  timestamp: string;
  preview: string;
}

interface ConversationSidebarProps {
  conversations: Conversation[];
  currentConversationId?: string;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  userName?: string;
}

export function ConversationSidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  userName = "Student",
}: ConversationSidebarProps) {
  const groupedConversations = {
    today: conversations.filter((c) => c.timestamp.includes("Today")),
    thisWeek: conversations.filter((c) => c.timestamp.includes("day ago")),
    older: conversations.filter(
      (c) => !c.timestamp.includes("Today") && !c.timestamp.includes("day ago")
    ),
  };

  return (
    <Sidebar>
      <SidebarHeader className="p-4">
        <Button
          onClick={onNewConversation}
          className="w-full justify-start gap-2"
        >
          <Plus className="h-4 w-4" />
          New Conversation
        </Button>
      </SidebarHeader>

      <SidebarContent>
        {groupedConversations.today.length > 0 && (
          <SidebarGroup>
            <SidebarGroupLabel>Today</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {groupedConversations.today.map((conversation) => (
                  <SidebarMenuItem key={conversation.id}>
                    <SidebarMenuButton
                      onClick={() => onSelectConversation(conversation.id)}
                      isActive={currentConversationId === conversation.id}
                      className="flex flex-col items-start h-auto py-3"
                    >
                      <div className="flex items-center gap-2 w-full">
                        <MessageSquare className="h-4 w-4 shrink-0" />
                        <span className="truncate">{conversation.title}</span>
                      </div>
                      <span className="text-xs text-muted-foreground pl-6 truncate w-full">
                        {conversation.preview}
                      </span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}

        {groupedConversations.thisWeek.length > 0 && (
          <SidebarGroup>
            <SidebarGroupLabel>This Week</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {groupedConversations.thisWeek.map((conversation) => (
                  <SidebarMenuItem key={conversation.id}>
                    <SidebarMenuButton
                      onClick={() => onSelectConversation(conversation.id)}
                      isActive={currentConversationId === conversation.id}
                      className="flex flex-col items-start h-auto py-3"
                    >
                      <div className="flex items-center gap-2 w-full">
                        <MessageSquare className="h-4 w-4 shrink-0" />
                        <span className="truncate">{conversation.title}</span>
                      </div>
                      <span className="text-xs text-muted-foreground pl-6 truncate w-full">
                        {conversation.preview}
                      </span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}

        {groupedConversations.older.length > 0 && (
          <SidebarGroup>
            <SidebarGroupLabel>Older</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {groupedConversations.older.map((conversation) => (
                  <SidebarMenuItem key={conversation.id}>
                    <SidebarMenuButton
                      onClick={() => onSelectConversation(conversation.id)}
                      isActive={currentConversationId === conversation.id}
                      className="flex flex-col items-start h-auto py-3"
                    >
                      <div className="flex items-center gap-2 w-full">
                        <MessageSquare className="h-4 w-4 shrink-0" />
                        <span className="truncate">{conversation.title}</span>
                      </div>
                      <span className="text-xs text-muted-foreground pl-6 truncate w-full">
                        {conversation.preview}
                      </span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
      </SidebarContent>

      <SidebarFooter className="p-4">
        <Separator className="mb-4" />
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
              <User className="h-4 w-4" />
            </div>
            <span className="text-sm">{userName}</span>
          </div>
          <Button variant="ghost" size="icon">
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
