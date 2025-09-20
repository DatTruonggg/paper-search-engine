"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Plus, MessageSquare, Search, FileText, MoreHorizontal, Trash2, Edit2 } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"

export interface ResearchSession {
  id: string
  title: string
  mode: "paper-finder" | "qa" | "summary"
  messageCount: number
  paperCount: number
  createdAt: Date
  lastActivity: Date
}

interface ResearchSessionSidebarProps {
  sessions: ResearchSession[]
  currentSessionId?: string
  onSessionSelect: (sessionId: string) => void
  onNewSession: () => void
  onDeleteSession: (sessionId: string) => void
  onRenameSession: (sessionId: string, newTitle: string) => void
}

export function ResearchSessionSidebar({
  sessions,
  currentSessionId,
  onSessionSelect,
  onNewSession,
  onDeleteSession,
  onRenameSession,
}: ResearchSessionSidebarProps) {
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState("")

  const handleStartEdit = (session: ResearchSession) => {
    setEditingSessionId(session.id)
    setEditTitle(session.title)
  }

  const handleSaveEdit = () => {
    if (editingSessionId && editTitle.trim()) {
      onRenameSession(editingSessionId, editTitle.trim())
    }
    setEditingSessionId(null)
    setEditTitle("")
  }

  const handleCancelEdit = () => {
    setEditingSessionId(null)
    setEditTitle("")
  }

  const getModeIcon = (mode: ResearchSession["mode"]) => {
    switch (mode) {
      case "paper-finder":
        return <Search className="h-4 w-4" />
      case "qa":
        return <MessageSquare className="h-4 w-4" />
      case "summary":
        return <FileText className="h-4 w-4" />
    }
  }

  const getModeLabel = (mode: ResearchSession["mode"]) => {
    switch (mode) {
      case "paper-finder":
        return "Paper Finder"
      case "qa":
        return "Q&A"
      case "summary":
        return "Summary"
    }
  }

  const formatDate = (date: Date) => {
    const now = new Date()
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60)

    if (diffInHours < 1) {
      return "Just now"
    } else if (diffInHours < 24) {
      return `${Math.floor(diffInHours)}h ago`
    } else if (diffInHours < 168) {
      return `${Math.floor(diffInHours / 24)}d ago`
    } else {
      return date.toLocaleDateString()
    }
  }

  return (
    <div className="flex flex-col h-full bg-gradient-to-b from-green-50 to-emerald-50/50">
      {/* Header */}
      <div className="p-4 border-b border-green-200 bg-white/80">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-green-800">Research Sessions</h2>
        </div>
        <Button
          onClick={onNewSession}
          className="w-full justify-start gap-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white shadow-sm transition-all"
        >
          <Plus className="h-4 w-4" />
          New Session
        </Button>
      </div>

      {/* Sessions List */}
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {sessions.map((session) => {
            const isActive = currentSessionId === session.id
            return (
              <div
                key={session.id}
                className={cn(
                  "group relative rounded-lg p-3 cursor-pointer transition-all duration-200",
                  "hover:bg-green-100/50 hover:shadow-sm",
                  isActive
                    ? "bg-gradient-to-r from-green-100 to-emerald-100 text-green-900 shadow-sm border border-green-200"
                    : "text-slate-700 hover:text-green-700",
                )}
                onClick={() => onSessionSelect(session.id)}
              >
                {editingSessionId === session.id ? (
                  <div className="space-y-2" onClick={(e) => e.stopPropagation()}>
                    <Input
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleSaveEdit()
                        if (e.key === "Escape") handleCancelEdit()
                      }}
                      onBlur={handleSaveEdit}
                      className="h-8 text-sm"
                      autoFocus
                    />
                  </div>
                ) : (
                  <>
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          {getModeIcon(session.mode)}
                          <span
                            className={cn(
                              "text-xs font-medium",
                              isActive ? "text-green-700" : "text-slate-600",
                            )}
                          >
                            {getModeLabel(session.mode)}
                          </span>
                        </div>
                        <h3 className="font-medium text-sm leading-tight mb-2 line-clamp-2">{session.title}</h3>
                        <div
                          className={cn(
                            "flex items-center gap-3 text-xs",
                            isActive ? "text-sidebar-accent-foreground/80" : "text-sidebar-foreground/70",
                          )}
                        >
                          <span>{session.messageCount} messages</span>
                          <span>{session.paperCount} papers</span>
                        </div>
                        <div
                          className={cn(
                            "text-xs mt-1",
                            isActive ? "text-sidebar-accent-foreground/80" : "text-sidebar-foreground/70",
                          )}
                        >
                          {formatDate(session.lastActivity)}
                        </div>
                      </div>

                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="opacity-0 group-hover:opacity-100 h-6 w-6 p-0 hover:bg-green-100"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <MoreHorizontal className="h-3 w-3" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-32">
                          <DropdownMenuItem onClick={() => handleStartEdit(session)}>
                            <Edit2 className="h-3 w-3 mr-2" />
                            Rename
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={() => onDeleteSession(session.id)}
                            className="text-destructive focus:text-destructive"
                          >
                            <Trash2 className="h-3 w-3 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </>
                )}
              </div>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}
