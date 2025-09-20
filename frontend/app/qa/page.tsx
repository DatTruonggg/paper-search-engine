"use client"

import { useEffect, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { ResearchSessionSidebar, type ResearchSession } from "@/components/research-session-sidebar"
import {
  MainSearchInterface,
  type SearchFilters,
  type QAMode,
} from "@/components/main-search-interface"
import { PaperResultsPanel, type PaperResult } from "@/components/paper-results-panel"
import { Menu, X, Search, MessageSquare, ArrowLeft } from "lucide-react"
import { ResearchService } from "@/lib/research-service"
import Link from "next/link"
import { cn } from "@/lib/utils"

type PersistedSearch = {
  contextId: string | null
  papers: PaperResult[]
  selectedIds: string[]
}

const STORAGE_KEY = "pse:lastSearch"

export default function QAPage() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sessions, setSessions] = useState<ResearchSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string>("1")
  const [searchResults, setSearchResults] = useState<PaperResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [lastQuestion, setLastQuestion] = useState<string>("")
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set())
  const [lastContextId, setLastContextId] = useState<string | null>(null)
  const api = useMemo(() => new ResearchService(), [])

  useEffect(() => {
    setSessions([
      {
        id: "1",
        title: "QA Session",
        mode: "qa",
        messageCount: 0,
        paperCount: 0,
        createdAt: new Date(),
        lastActivity: new Date(),
      },
    ])
  }, [])

  // Load persisted search results/context from Search page
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (!raw) return
      const persisted: PersistedSearch = JSON.parse(raw)
      setSearchResults(persisted?.papers || [])
      setSelectedPapers(new Set(persisted?.selectedIds || []))
      setLastContextId(persisted?.contextId ?? null)
    } catch {}
  }, [])

  const currentSession = sessions.find((s) => s.id === currentSessionId)

  const handleNewSession = () => {
    const newSession: ResearchSession = {
      id: Date.now().toString(),
      title: "New QA Session",
      mode: "qa",
      messageCount: 0,
      paperCount: searchResults.length,
      createdAt: new Date(),
      lastActivity: new Date(),
    }
    setSessions((prev) => [newSession, ...prev])
    setCurrentSessionId(newSession.id)
    setSidebarOpen(false)
  }

  const handleSessionSelect = (sessionId: string) => {
    setCurrentSessionId(sessionId)
    setSidebarOpen(false)
  }

  const handleDeleteSession = (sessionId: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== sessionId))
    if (currentSessionId === sessionId) {
      const remaining = sessions.filter((s) => s.id !== sessionId)
      if (remaining.length > 0) setCurrentSessionId(remaining[0].id)
      else handleNewSession()
    }
  }

  const handleRenameSession = (sessionId: string, newTitle: string) => {
    setSessions((prev) => prev.map((s) => (s.id === sessionId ? { ...s, title: newTitle } : s)))
  }

  // QA submission: if single-paper and none selected -> disabled by component; if all-papers -> use contextId
  const handleAsk = async (question: string, _filters?: SearchFilters, qaMode?: QAMode) => {
    setIsLoading(true)
    setLastQuestion(question)
    try {
      if (qaMode === "single-paper") {
        const paperIds = Array.from(selectedPapers)
        if (paperIds.length === 0) return
        await api.qaSelected({ question, paperIds, citationStyle: "APA" })
      } else {
        if (!lastContextId) throw new Error("No search context available. Perform a search first.")
        await api.qaAll({ question, searchContextId: lastContextId, citationStyle: "APA" })
      }

      if (currentSession) {
        setSessions((prev) =>
          prev.map((s) =>
            s.id === currentSessionId
              ? { ...s, messageCount: s.messageCount + 1, lastActivity: new Date(), title: s.title }
              : s,
          ),
        )
      }
    } catch (e) {
      console.error("[qa] error:", e)
    } finally {
      setIsLoading(false)
    }
  }

  const handlePaperSelectionChange = (newSelection: Set<string>) => {
    setSelectedPapers(newSelection)
    // persist selection back into the same key so it stays in sync with Search
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      const base: PersistedSearch = raw ? JSON.parse(raw) : { contextId: lastContextId, papers: searchResults, selectedIds: [] }
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ ...base, selectedIds: Array.from(newSelection) }),
      )
    } catch {}
  }

  const handleBookmarkToggle = async (paperId: string) => {
    setSearchResults((prev) => prev.map((p) => (p.id === paperId ? { ...p, isBookmarked: !p.isBookmarked } : p)))
    try {
      const on = !searchResults.find((p) => p.id === paperId)?.isBookmarked
      await api.bookmark(paperId, on ?? true)
    } catch (e) {
      setSearchResults((prev) => prev.map((p) => (p.id === paperId ? { ...p, isBookmarked: !p.isBookmarked } : p)))
      console.error(e)
    }
  }

  const handleViewFullPaper = (paper: PaperResult) => {
    window.open(paper.url, "_blank")
  }

  return (
    <div className="flex h-screen bg-background">
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      <div
        className={`
        fixed lg:static inset-y-0 left-0 z-50 w-80
        bg-gradient-to-br from-slate-50 via-white to-slate-100
        border-r-2 border-gradient-to-b from-purple-200 via-blue-200 to-indigo-200
        shadow-2xl shadow-purple-100/50
        transform transition-all duration-300 ease-in-out lg:transform-none
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
      `}
      >
        {/* Mobile Header */}
        <div className="flex items-center justify-between p-6 border-b-2 border-gradient-to-r from-purple-200 via-blue-200 to-indigo-200 lg:hidden bg-gradient-to-r from-purple-50 via-blue-50 to-indigo-50 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-sm">💬</span>
            </div>
            <h2 className="text-xl font-extrabold bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">Chat QA</h2>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(false)}
            className="hover:bg-gradient-to-r hover:from-purple-100 hover:to-blue-100 transition-all duration-300 rounded-xl border border-purple-200"
          >
            <X className="h-5 w-5 text-purple-600" />
          </Button>
        </div>

        {/* Sidebar Content */}
        <div className="p-6 space-y-6 h-full overflow-y-auto">
          {/* Welcome Section */}
          <div className="text-center pb-4 border-b border-gradient-to-r from-purple-100 to-blue-100">
            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 rounded-2xl flex items-center justify-center shadow-2xl rotate-3 hover:rotate-0 transition-transform duration-500">
              <span className="text-3xl">💬</span>
            </div>
            <h3 className="text-lg font-bold bg-gradient-to-r from-indigo-700 via-purple-700 to-pink-700 bg-clip-text text-transparent">QA Assistant</h3>
            <p className="text-xs text-slate-600 mt-1">Ask questions about papers</p>
          </div>

          {/* Back to Search Button - Enhanced */}
          <Button asChild className="w-full justify-start gap-4 h-14 bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 hover:from-emerald-600 hover:via-teal-600 hover:to-cyan-600 text-white shadow-2xl hover:shadow-emerald-300/50 transition-all duration-500 rounded-2xl font-bold text-base group overflow-hidden relative">
            <Link href="/search">
              <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/10 to-white/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
              <div className="p-2 bg-white/20 rounded-xl">
                <Search className="h-6 w-6 group-hover:scale-110 transition-transform duration-300" />
              </div>
              <span className="group-hover:translate-x-1 transition-transform duration-300">🔍 Paper Search</span>
            </Link>
          </Button>

          {/* Session Management */}
          <div className="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 border-2 border-blue-200 rounded-2xl p-4 shadow-xl hover:shadow-blue-200/50 transition-all duration-300">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-sm">📝</span>
              </div>
              <span className="text-base font-bold bg-gradient-to-r from-blue-700 to-indigo-700 bg-clip-text text-transparent">Current Session</span>
            </div>
            <div className="bg-white/70 rounded-xl p-3 border border-blue-200/50">
              <p className="text-sm font-semibold text-blue-800">{currentSession?.title || "QA Session"}</p>
              <div className="flex justify-between items-center mt-2 text-xs text-slate-600">
                <span>Messages: {currentSession?.messageCount || 0}</span>
                <span>Papers: {searchResults.length}</span>
              </div>
            </div>
            <Button
              onClick={handleNewSession}
              className="w-full mt-3 bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white shadow-lg rounded-xl font-semibold text-sm transition-all duration-300"
            >
              ✨ New Session
            </Button>
          </div>

          {/* Paper Context Info */}
          <div className="bg-gradient-to-br from-amber-50 via-orange-50 to-red-50 border-2 border-amber-200 rounded-2xl p-4 shadow-xl">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-gradient-to-br from-amber-500 to-orange-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-sm">📊</span>
              </div>
              <span className="text-base font-bold bg-gradient-to-r from-amber-700 to-orange-700 bg-clip-text text-transparent">Paper Context</span>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Available Papers:</span>
                <span className="font-bold text-amber-700">{searchResults.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Selected:</span>
                <span className="font-bold text-amber-700">{selectedPapers.size}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Context:</span>
                <span className={cn("font-bold text-xs", lastContextId ? "text-green-600" : "text-red-500")}>
                  {lastContextId ? "Ready ✓" : "None"}
                </span>
              </div>
            </div>
            {!lastContextId && (
              <div className="mt-3 p-2 bg-red-100 border border-red-200 rounded-lg">
                <p className="text-xs text-red-700 font-medium">⚠️ No search context found. Perform a search first.</p>
              </div>
            )}
          </div>

          {/* QA Tips */}
          <div className="bg-gradient-to-br from-violet-50 via-fuchsia-50 to-pink-50 border-2 border-violet-200 rounded-2xl p-4 shadow-xl">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-sm">💡</span>
              </div>
              <span className="text-base font-bold bg-gradient-to-r from-violet-700 to-pink-700 bg-clip-text text-transparent">QA Tips</span>
            </div>
            <div className="space-y-2 text-xs text-slate-700">
              <p>• Ask specific questions about the research</p>
              <p>• Select papers for focused analysis</p>
              <p>• Use "All Papers" for broader insights</p>
              <p>• Try questions like "What are the key findings?"</p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col lg:flex-row min-w-0">
        <div className="lg:hidden flex items-center justify-between p-4 border-b border-border bg-card">
          <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(true)}>
            <Menu className="h-4 w-4" />
          </Button>
          <h1 className="font-semibold text-foreground">QA Session</h1>
          <div className="w-8" />
        </div>

        <div className="flex-1 min-w-0">
          <MainSearchInterface
            mode="qa"
            forceMode="qa"
            hideModeSelector
            onModeChange={() => {}}
            onSearch={handleAsk}
            isLoading={isLoading}
            selectedPapers={selectedPapers}
            totalPapers={searchResults.length}
          />
        </div>

        <div className="w-full lg:w-96 border-t lg:border-t-0 lg:border-l border-border">
          <PaperResultsPanel
            papers={searchResults}
            isLoading={isLoading}
            onBookmarkToggle={handleBookmarkToggle}
            onViewFullPaper={handleViewFullPaper}
            searchQuery={lastQuestion}
            mode="qa"
            selectedPapers={selectedPapers}
            onPaperSelectionChange={handlePaperSelectionChange}
          />
        </div>
      </div>
    </div>
  )
}


