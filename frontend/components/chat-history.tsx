"use client"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Plus, MessageSquare } from "lucide-react"
import type { ChatSession } from "@/app/page"
import { cn } from "@/lib/utils"

interface ChatHistoryProps {
  sessions: ChatSession[]
  currentSessionId: string
  onSessionSelect: (sessionId: string) => void
  onNewSession: () => void
}

export function ChatHistory({ sessions, currentSessionId, onSessionSelect, onNewSession }: ChatHistoryProps) {
  return (
    <div className="flex flex-col h-full">
      <div className="p-4">
        <Button
          onClick={onNewSession}
          className="w-full justify-start gap-2 bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
        >
          <Plus className="h-4 w-4" />
          New Research Session
        </Button>
      </div>

      <ScrollArea className="flex-1 px-4">
        <div className="space-y-2 pb-4">
          {sessions.map((session) => (
            <Button
              key={session.id}
              variant="ghost"
              className={cn(
                "w-full justify-start gap-2 h-auto p-3 text-left",
                "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                currentSessionId === session.id && "bg-sidebar-accent text-sidebar-accent-foreground",
              )}
              onClick={() => onSessionSelect(session.id)}
            >
              <MessageSquare className="h-4 w-4 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="font-medium text-sm truncate">{session.title}</div>
                <div className="text-xs text-muted-foreground mt-1">{session.messages.length} messages</div>
                <div className="text-xs text-muted-foreground">{session.createdAt.toLocaleDateString()}</div>
              </div>
            </Button>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}
