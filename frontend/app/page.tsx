"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { ResearchSessionSidebar, type ResearchSession } from "@/components/research-session-sidebar"
import {
  MainSearchInterface,
  type SearchMode,
  type SearchFilters,
  type QAMode,
} from "@/components/main-search-interface"
import { PaperResultsPanel, type PaperResult } from "@/components/paper-results-panel"
import { Menu, X } from "lucide-react"
import { ResearchService } from "@/lib/research-service"

const mockPapers: PaperResult[] = [
  {
    id: "1",
    title: "Attention Is All You Need",
    authors: [
      "Vaswani, A.",
      "Shazeer, N.",
      "Parmar, N.",
      "Uszkoreit, J.",
      "Jones, L.",
      "Gomez, A. N.",
      "Kaiser, L.",
      "Polosukhin, I.",
    ],
    abstract:
      "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    citationCount: 85000,
    publicationDate: "2017-06-12",
    venue: "NIPS 2017",
    doi: "10.48550/arXiv.1706.03762",
    url: "https://arxiv.org/abs/1706.03762",
    keywords: ["attention", "transformer", "neural networks", "sequence modeling"],
    pdfUrl: "https://arxiv.org/pdf/1706.03762.pdf",
    isBookmarked: false,
  },
  {
    id: "2",
    title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    authors: ["Devlin, J.", "Chang, M. W.", "Lee, K.", "Toutanova, K."],
    abstract:
      "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
    citationCount: 65000,
    publicationDate: "2018-10-11",
    venue: "NAACL 2019",
    doi: "10.48550/arXiv.1810.04805",
    url: "https://arxiv.org/abs/1810.04805",
    keywords: ["BERT", "language model", "bidirectional", "pre-training"],
    pdfUrl: "https://arxiv.org/pdf/1810.04805.pdf",
    isBookmarked: true,
  },
  {
    id: "3",
    title: "GPT-3: Language Models are Few-Shot Learners",
    authors: ["Brown, T. B.", "Mann, B.", "Ryder, N.", "Subbiah, M.", "Kaplan, J.", "Dhariwal, P."],
    abstract:
      "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples.",
    citationCount: 45000,
    publicationDate: "2020-05-28",
    venue: "NeurIPS 2020",
    doi: "10.48550/arXiv.2005.14165",
    url: "https://arxiv.org/abs/2005.14165",
    keywords: ["GPT-3", "few-shot learning", "language model", "scaling"],
    pdfUrl: "https://arxiv.org/pdf/2005.14165.pdf",
    isBookmarked: false,
  },
]

export default function PaperSearchChatbot() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sessions, setSessions] = useState<ResearchSession[]>([
    {
      id: "1",
      title: "Transformer Architecture Research",
      mode: "paper-finder",
      messageCount: 5,
      paperCount: 12,
      createdAt: new Date(Date.now() - 86400000),
      lastActivity: new Date(Date.now() - 3600000),
    },
    {
      id: "2",
      title: "Language Model Q&A Session",
      mode: "qa",
      messageCount: 8,
      paperCount: 6,
      createdAt: new Date(Date.now() - 172800000),
      lastActivity: new Date(Date.now() - 7200000),
    },
  ])
  const [currentSessionId, setCurrentSessionId] = useState<string>("1")
  const [currentMode, setCurrentMode] = useState<SearchMode>("paper-finder")
  const [searchResults, setSearchResults] = useState<PaperResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [lastSearchQuery, setLastSearchQuery] = useState<string>("")
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set())
  const [lastContextId, setLastContextId] = useState<string | null>(null)
  const api = new ResearchService()

  const currentSession = sessions.find((s) => s.id === currentSessionId)

  const handleNewSession = () => {
    const newSession: ResearchSession = {
      id: Date.now().toString(),
      title: "New Research Session",
      mode: currentMode,
      messageCount: 0,
      paperCount: 0,
      createdAt: new Date(),
      lastActivity: new Date(),
    }

    setSessions((prev) => [newSession, ...prev])
    setCurrentSessionId(newSession.id)
    setSearchResults([])
    setSelectedPapers(new Set())
    setSidebarOpen(false)
  }

  const handleSessionSelect = (sessionId: string) => {
    setCurrentSessionId(sessionId)
    const session = sessions.find((s) => s.id === sessionId)
    if (session) {
      setCurrentMode(session.mode)
      if (session.mode === "paper-finder") {
        setSearchResults(mockPapers.slice(0, session.paperCount))
      } else {
        setSearchResults([])
      }
    }
    setSelectedPapers(new Set())
    setSidebarOpen(false)
  }

  const handleDeleteSession = (sessionId: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== sessionId))
    if (currentSessionId === sessionId) {
      const remainingSessions = sessions.filter((s) => s.id !== sessionId)
      if (remainingSessions.length > 0) {
        setCurrentSessionId(remainingSessions[0].id)
      } else {
        handleNewSession()
      }
    }
  }

  const handleRenameSession = (sessionId: string, newTitle: string) => {
    setSessions((prev) => prev.map((s) => (s.id === sessionId ? { ...s, title: newTitle } : s)))
  }

  const handleModeChange = (mode: SearchMode) => {
    setCurrentMode(mode)
    if (currentSession) {
      setSessions((prev) => prev.map((s) => (s.id === currentSessionId ? { ...s, mode, lastActivity: new Date() } : s)))
    }
    setSearchResults([])
    setSelectedPapers(new Set())
  }

  const handleSearch = async (query: string, filters?: SearchFilters, qaMode?: QAMode) => {
    setIsLoading(true)
    setLastSearchQuery(query)

    try {
      if (currentMode === "paper-finder") {
        const resp = await api.searchPapers(query, filters, 1, 20, "relevance")
        setSearchResults(resp.papers)
        setLastContextId(resp.contextId)
      } else if (currentMode === "qa") {
        // Placeholder: actual QA call should hit /v1/qa with mode selected/all and use lastContextId
        setSearchResults(searchResults)
      } else if (currentMode === "summary") {
        setSearchResults(searchResults)
      }

      if (currentSession) {
        setSessions((prev) =>
          prev.map((s) =>
            s.id === currentSessionId
              ? {
                  ...s,
                  messageCount: s.messageCount + 1,
                  paperCount: (currentMode === "paper-finder" ? searchResults.length : s.paperCount),
                  lastActivity: new Date(),
                  title: s.title === "New Research Session" ? query.slice(0, 50) + "..." : s.title,
                }
              : s,
          ),
        )
      }
    } catch (error) {
      console.error("[v0] Search error:", error)
      setSearchResults([])
    } finally {
      setIsLoading(false)
    }
  }

  const handlePaperSelect = (paper: PaperResult) => {
    console.log("[v0] Paper selected:", paper.title)
  }

  const handlePaperSelectionChange = (newSelection: Set<string>) => {
    setSelectedPapers(newSelection)
    console.log(`[v0] Paper selection updated: ${newSelection.size} papers selected`)
  }

  const handleBookmarkToggle = async (paperId: string) => {
    setSearchResults((prev) => prev.map((p) => (p.id === paperId ? { ...p, isBookmarked: !p.isBookmarked } : p)))
    try {
      const on = !searchResults.find((p) => p.id === paperId)?.isBookmarked
      await api.bookmark(paperId, on ?? true)
    } catch (e) {
      // revert on failure
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
        fixed lg:static inset-y-0 left-0 z-50 w-80 bg-sidebar border-r border-sidebar-border
        transform transition-transform duration-200 ease-in-out lg:transform-none
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
      `}
      >
        <div className="flex items-center justify-between p-4 border-b border-sidebar-border lg:hidden">
          <h2 className="text-lg font-semibold text-sidebar-foreground">Research Sessions</h2>
          <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(false)}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <ResearchSessionSidebar
          sessions={sessions}
          currentSessionId={currentSessionId}
          onSessionSelect={handleSessionSelect}
          onNewSession={handleNewSession}
          onDeleteSession={handleDeleteSession}
          onRenameSession={handleRenameSession}
        />
      </div>

      <div className="flex-1 flex flex-col lg:flex-row min-w-0">
        <div className="lg:hidden flex items-center justify-between p-4 border-b border-border bg-card">
          <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(true)}>
            <Menu className="h-4 w-4" />
          </Button>
          <h1 className="font-semibold text-foreground">{currentSession?.title || "Research Session"}</h1>
          <div className="w-8" />
        </div>

        <div className="flex-1 min-w-0">
          <MainSearchInterface
            mode={currentMode}
            onModeChange={handleModeChange}
            onSearch={handleSearch}
            isLoading={isLoading}
            selectedPapers={selectedPapers}
            totalPapers={searchResults.length}
          />
        </div>

        <div className="w-full lg:w-96 border-t lg:border-t-0 lg:border-l border-border">
          <PaperResultsPanel
            papers={searchResults}
            isLoading={isLoading}
            onPaperSelect={handlePaperSelect}
            onBookmarkToggle={handleBookmarkToggle}
            onViewFullPaper={handleViewFullPaper}
            searchQuery={lastSearchQuery}
            mode={currentMode}
            selectedPapers={selectedPapers}
            onPaperSelectionChange={handlePaperSelectionChange}
          />
        </div>
      </div>
    </div>
  )
}
