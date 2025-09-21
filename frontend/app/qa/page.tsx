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
import { ChatMessage, type MessageWithCitations } from "@/components/chat-message"
import { Input } from "@/components/ui/input"
import { Menu, X, Search, MessageSquare, ArrowLeft, Bot } from "lucide-react"
import { ResearchService, type QAResponse } from "@/lib/research-service"
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
  const [qaResult, setQaResult] = useState<QAResponse | null>(null)
  const [qaError, setQaError] = useState<string | null>(null)
  const [messages, setMessages] = useState<MessageWithCitations[]>([])
  const [chatInput, setChatInput] = useState<string>("")
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
    // Reset conversation state to show the first QA screen
    setMessages([])
    setChatInput("")
    setQaResult(null)
    setQaError(null)
    setLastQuestion("")
    setSelectedPapers(new Set())
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

  // Làm sạch answer: loại bỏ tham chiếu (Chunk x, Figure y, ...)
  const cleanAnswerText = (text: string): string => {
    if (!text) return text
    let out = text
    // Xóa các ngoặc đơn chỉ chứa tham chiếu Chunk/Figure/Table/Image (có thể nhiều mục, cách nhau bởi dấu phẩy)
    out = out.replace(/\s*\((?:\s*(?:Chunk\s*\d+|Figure\s*\d+|Fig\.?\s*\d+|Table\s*\d+|Image\s*\d+))(?:\s*,\s*(?:Chunk\s*\d+|Figure\s*\d+|Fig\.?\s*\d+|Table\s*\d+|Image\s*\d+))*\)\s*/gi, ' ')
    // Thu gọn khoảng trắng thừa
    out = out.replace(/\s{2,}/g, ' ')
    // Xóa khoảng trắng trước dấu câu
    out = out.replace(/\s+([.,;:!?])/g, '$1')
    return out.trim()
  }

  // Helper: map QAResponse -> Assistant Chat Message
  const mapQAtoAssistant = (resp: QAResponse): MessageWithCitations => {
    const sources = (resp.sources || []).map((s: any, idx: number) => ({
      id: (s?.paper_id || s?.chunk_index || String(idx)).toString(),
      title: s?.title || s?.section_path || s?.paper_id || `Source ${idx + 1}`,
      authors: Array.isArray(s?.authors) ? s.authors : [],
      url: s?.url || s?.minio_pdf_url,
      type: "paper" as const,
    }))
    return {
      id: `assistant_${Date.now()}`,
      role: "assistant",
      content: cleanAnswerText(resp.answer),
      timestamp: new Date(),
      sources,
    }
  }

  // Send QA question using selected papers (or all current results for multi mode)
  const handleAsk = async (question: string, _filters?: SearchFilters, qaMode?: QAMode) => {
    setIsLoading(true)
    setLastQuestion(question)
    setQaError(null)
    try {
      const selectedIds = Array.from(selectedPapers)
      // Log payload and quick metadata for debugging backend errors like `'paper_title'`
      const selectedMeta = selectedIds.map((id) => {
        const p = searchResults.find((x) => x.id === id)
        return { id, title: p?.title }
      })
      console.log('[QA] ask:start', { mode: qaMode || 'qa', question, selectedIds, selectedMeta })
      const userMsg: MessageWithCitations = {
        id: `user_${Date.now()}`,
        role: "user",
        content: question,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, userMsg])
      if (qaMode === "single-paper") {
        if (selectedIds.length === 0) return
        const firstOnly = [selectedIds[0]]
        console.log('[QA] POST /api/v1/qa/(single|multi)-paper payload', { paper_ids: selectedIds, count: selectedIds.length, question })
        const resp = await api.qaSelected({ question, paperIds: selectedIds, citationStyle: "APA" })
        console.log('[QA] response', {
          confidence: Math.round((resp.confidenceScore || 0) * 100),
          chunks: resp.contextChunksCount,
          papersInvolved: resp.papersInvolved,
        })
        setMessages((prev) => [...prev, mapQAtoAssistant(resp)])
      } else {
        // multi-paper: dùng danh sách chọn; nếu rỗng, fallback toàn bộ kết quả hiện có
        const ids = selectedIds.length > 0 ? selectedIds : searchResults.map(p => p.id)
        if (ids.length === 0) throw new Error("No papers available for multi-paper QA")
        console.log('[QA] POST /api/v1/qa/multi-paper payload', { paper_ids: ids, count: ids.length, question })
        const resp = await api.qaSelected({ question, paperIds: ids, citationStyle: "APA" })
        console.log('[QA] response', {
          confidence: Math.round((resp.confidenceScore || 0) * 100),
          chunks: resp.contextChunksCount,
          papersInvolved: resp.papersInvolved,
        })
        setMessages((prev) => [...prev, mapQAtoAssistant(resp)])
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
      console.error("[QA] error", e)
      setQaError((e as Error)?.message || "Unknown error")
    } finally {
      setIsLoading(false)
    }
  }

  // Chat follow-up messages after the first answer
  const handleChatSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    const text = chatInput.trim()
    if (!text || isLoading) return
    setIsLoading(true)
    setQaError(null)
    const userMsg: MessageWithCitations = {
      id: `user_${Date.now()}`,
      role: "user",
      content: text,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMsg])
    setChatInput("")
    try {
      const ids = Array.from(selectedPapers)
      const paperIds = ids.length > 0 ? ids : searchResults.map(p => p.id)
      if (paperIds.length === 0) throw new Error("No papers available for QA")
      const meta = paperIds.map((id) => ({ id, title: searchResults.find(x => x.id === id)?.title }))
      console.log('[QA] chat:start', { question: text, paperIds, meta })
      const resp = await api.qaSelected({ question: text, paperIds, citationStyle: "APA" })
      console.log('[QA] chat:response', {
        confidence: Math.round((resp.confidenceScore || 0) * 100),
        chunks: resp.contextChunksCount,
        papersInvolved: resp.papersInvolved,
      })
      setMessages((prev) => [...prev, mapQAtoAssistant(resp)])
    } catch (e) {
      console.error('[QA] chat:error', e)
      setQaError((e as Error)?.message || "Unknown error")
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

    // Fetch authoritative metadata for selected papers (batch endpoint).
    // This ensures we have updated title/abstract/links before QA ask.
    ;(async () => {
      if (newSelection.size === 0) return
      try {
        const { papers: batchPapers } = await api.fetchPapersBatch(Array.from(newSelection))
        if (!batchPapers.length) return
        setSearchResults(prev => {
          const byId = new Map(prev.map(p => [p.id, p]))
            batchPapers.forEach(bp => {
              const existing = byId.get(bp.id)
              if (existing) {
                // Merge keeping any UI state like bookmark
                byId.set(bp.id, { ...existing, ...bp, isBookmarked: existing.isBookmarked })
              } else {
                byId.set(bp.id, bp)
              }
            })
          return Array.from(byId.values())
        })
      } catch (e) {
        console.error('[qa] batch metadata fetch failed', e)
      }
    })()
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
        border-r-2 border-gradient-to-b from-green-200 via-emerald-200 to-green-300
        shadow-2xl shadow-green-100/50
        transform transition-all duration-300 ease-in-out lg:transform-none
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
      `}
      >
        {/* Mobile Header */}
        <div className="flex items-center justify-between p-6 border-b-2 border-gradient-to-r from-green-200 via-emerald-200 to-green-300 lg:hidden bg-gradient-to-r from-green-50 via-emerald-50 to-green-50 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-sm">💬</span>
            </div>
            <h2 className="text-xl font-extrabold bg-gradient-to-r from-green-600 via-emerald-600 to-green-700 bg-clip-text text-transparent">Chat QA</h2>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(false)}
            className="hover:bg-gradient-to-r hover:from-green-100 hover:to-emerald-100 transition-all duration-300 rounded-xl border border-green-200"
          >
            <X className="h-5 w-5 text-green-600" />
          </Button>
        </div>

        {/* Sidebar Content */}
        <div className="p-6 space-y-6 h-full overflow-y-auto">
          {/* Welcome Section */}
          <div className="text-center pb-4 border-b border-gradient-to-r from-green-100 to-emerald-100">
            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-green-500 via-emerald-500 to-green-600 rounded-2xl flex items-center justify-center shadow-2xl rotate-3 hover:rotate-0 transition-transform duration-500">
              <span className="text-3xl">💬</span>
            </div>
            <h3 className="text-lg font-bold bg-gradient-to-r from-green-700 via-emerald-700 to-green-800 bg-clip-text text-transparent">QA Assistant</h3>
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
          <div className="bg-gradient-to-br from-green-50 via-emerald-50 to-green-50 border-2 border-green-200 rounded-2xl p-4 shadow-xl hover:shadow-green-200/50 transition-all duration-300">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-sm">📝</span>
              </div>
              <span className="text-base font-bold bg-gradient-to-r from-green-700 to-emerald-700 bg-clip-text text-transparent">Current Session</span>
            </div>
            <div className="bg-white/70 rounded-xl p-3 border border-green-200/50">
              <p className="text-sm font-semibold text-green-800">{currentSession?.title || "QA Session"}</p>
              <div className="flex justify-between items-center mt-2 text-xs text-slate-600">
                <span>Messages: {currentSession?.messageCount || 0}</span>
                <span>Papers: {searchResults.length}</span>
              </div>
            </div>
            <Button
              onClick={handleNewSession}
              className="w-full mt-3 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white shadow-lg rounded-xl font-semibold text-sm transition-all duration-300"
            >
              ✨ New Session
            </Button>
          </div>

          {/* Paper Context Info */}
          <div className="bg-gradient-to-br from-green-50 via-emerald-50 to-green-50 border-2 border-green-200 rounded-2xl p-4 shadow-xl">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-sm">📊</span>
              </div>
              <span className="text-base font-bold bg-gradient-to-r from-green-700 to-emerald-700 bg-clip-text text-transparent">Paper Context</span>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Available Papers:</span>
                <span className="font-bold text-green-700">{searchResults.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Selected:</span>
                <span className="font-bold text-green-700">{selectedPapers.size}</span>
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
          <div className="bg-gradient-to-br from-green-50 via-emerald-50 to-green-50 border-2 border-green-200 rounded-2xl p-4 shadow-xl">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-sm">💡</span>
              </div>
              <span className="text-base font-bold bg-gradient-to-r from-green-700 to-emerald-700 bg-clip-text text-transparent">QA Tips</span>
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
          {/* Nếu chưa có hội thoại, hiển thị ô hỏi ban đầu; sau đó chuyển sang chat UI */}
          {messages.length === 0 ? (
            <>
              {qaError && (
                <div className="m-4 p-3 rounded-lg border border-red-200 bg-red-50 text-red-700 text-sm">{qaError}</div>
              )}
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
            </>
          ) : (
            <div className="flex h-full flex-col">
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((m) => (
                  <ChatMessage key={m.id} message={m} />
                ))}
                {isLoading && (
                  <div className="flex gap-3 justify-start">
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                        <Bot className="h-4 w-4 text-primary-foreground" />
                      </div>
                    </div>
                    <div className="max-w-[80%]">
                      <div className="p-4 bg-card text-card-foreground rounded-md border border-border">
                        <div className="flex items-center gap-1 h-4" aria-label="Thinking...">
                          <span className="w-2 h-2 rounded-full bg-muted-foreground/60 animate-bounce" style={{ animationDelay: '0ms' }} />
                          <span className="w-2 h-2 rounded-full bg-muted-foreground/60 animate-bounce" style={{ animationDelay: '150ms' }} />
                          <span className="w-2 h-2 rounded-full bg-muted-foreground/60 animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              {qaError && (
                <div className="mx-4 mb-2 p-2 rounded border border-red-200 bg-red-50 text-red-700 text-xs">{qaError}</div>
              )}
              <form onSubmit={handleChatSend} className="p-4 border-t border-border flex gap-2">
                <Input
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Type your next question..."
                  disabled={isLoading}
                />
                <Button type="submit" disabled={!chatInput.trim() || isLoading}>
                  Send
                </Button>
              </form>
            </div>
          )}
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


