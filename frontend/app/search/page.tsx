"use client"

import { useEffect, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import {
  MainSearchInterface,
  type SearchFilters,
  type QAMode,
} from "@/components/main-search-interface"
import { PaperResultsPanel, type PaperResult } from "@/components/paper-results-panel"
import { Menu, X, Bot, MessageSquare } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { ResearchService } from "@/lib/research-service"
import Link from "next/link"
import { useToast } from "@/hooks/use-toast"
import { ErrorBoundary } from "@/components/error-boundary"
import { OfflineIndicator } from "@/components/offline-indicator"

type PersistedSearch = {
  contextId: string | null
  papers: PaperResult[]
  selectedIds: string[]
}

const STORAGE_KEY = "pse:lastSearch"

export default function SearchPage() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [searchResults, setSearchResults] = useState<PaperResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [lastSearchQuery, setLastSearchQuery] = useState<string>("")
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set())
  const [lastContextId, setLastContextId] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalCount, setTotalCount] = useState(0)
  const [paginationMode, setPaginationMode] = useState<'pagination' | 'load-more'>('pagination')
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [agentMode, setAgentMode] = useState(false)
  const pageSize = 20
  const api = useMemo(() => new ResearchService(), [])
  const { toast } = useToast()

  // Load persisted search context
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (!raw) return
      const persisted: PersistedSearch = JSON.parse(raw)
      if (persisted?.papers) {
        setSearchResults(persisted.papers)
      }
      if (persisted?.selectedIds) {
        setSelectedPapers(new Set(persisted.selectedIds))
      }
      setLastContextId(persisted?.contextId ?? null)
    } catch {}
  }, [])

  const persist = (updates: Partial<PersistedSearch>) => {
    try {
      const current: PersistedSearch = {
        contextId: lastContextId ?? null,
        papers: searchResults,
        selectedIds: Array.from(selectedPapers),
      }
      const next = { ...current, ...updates }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    } catch {}
  }

  const handleSearch = async (query: string, filters?: SearchFilters, _qaMode?: QAMode, isAgentMode?: boolean, page = 1) => {
    setIsLoading(true)
    setLastSearchQuery(query)
    if (page === 1) {
      setCurrentPage(1)
      setSearchResults([])
      setSelectedPapers(new Set())
    }
    try {
      // Use agent search if enabled
      const resp = isAgentMode
        ? await api.searchPapersWithAgent(query, filters, page, pageSize, "relevance")
        : await api.searchPapers(query, filters, page, pageSize, "relevance")

      if (page === 1) {
        setSearchResults(resp.papers)
      } else {
        setSearchResults(prev => [...prev, ...resp.papers])
      }

      setLastContextId(resp.contextId)
      setCurrentPage(page)
      setTotalCount(resp.total)

      // Show success toast for new searches
      if (page === 1 && resp.papers.length > 0) {
        toast({
          title: isAgentMode ? "AI Agent search completed" : "Search completed",
          description: `Found ${resp.total} papers${isAgentMode ? ' using intelligent analysis' : ''}.`,
          duration: 3000,
        })
      }

      persist({ contextId: resp.contextId, papers: page === 1 ? resp.papers : [...searchResults, ...resp.papers], selectedIds: [] })
    } catch (e) {
      console.error("[search] error:", e)
      if (page === 1) {
        setSearchResults([])
        setLastContextId(null)
        setTotalCount(0)
        persist({ contextId: null, papers: [], selectedIds: [] })

        toast({
          title: "Search failed",
          description: "Unable to complete the search. Please try again.",
          variant: "destructive",
          duration: 5000,
        })
      }
    } finally {
      setIsLoading(false)
      setIsLoadingMore(false)
    }
  }

  const handlePageChange = (page: number) => {
    handleSearch(lastSearchQuery, undefined, undefined, false, page)
  }

  const handleLoadMore = async () => {
    if (isLoadingMore || isLoading) return
    setIsLoadingMore(true)
    await handleSearch(lastSearchQuery, undefined, undefined, false, currentPage + 1)
  }

  const handleSuggestionSearch = (suggestion: string) => {
    handleSearch(suggestion, undefined, undefined, false, 1)
  }

  const handleClearFilters = () => {
    // This would be implemented to clear any active filters
    // For now, we'll just trigger a new search with the current query
    if (lastSearchQuery) {
      handleSearch(lastSearchQuery, undefined, undefined, false, 1)
    }
  }

  const handlePaperSelect = (paper: PaperResult) => {
    console.log("[search] paper selected:", paper.title)
  }

  const handlePaperSelectionChange = (newSelection: Set<string>) => {
    setSelectedPapers(newSelection)
    persist({ selectedIds: Array.from(newSelection) })
  }

  const handleBookmarkToggle = async (paperId: string) => {
    const paper = searchResults.find((p) => p.id === paperId)
    const wasBookmarked = paper?.isBookmarked

    // Optimistically update UI
    setSearchResults((prev) => prev.map((p) => (p.id === paperId ? { ...p, isBookmarked: !p.isBookmarked } : p)))

    try {
      const on = !wasBookmarked
      await api.bookmark(paperId, on ?? true)

      toast({
        title: on ? "Paper bookmarked" : "Bookmark removed",
        description: on
          ? `"${paper?.title}" has been added to your bookmarks.`
          : `"${paper?.title}" has been removed from your bookmarks.`,
        duration: 3000,
      })
    } catch (e) {
      // Revert optimistic update on error
      setSearchResults((prev) => prev.map((p) => (p.id === paperId ? { ...p, isBookmarked: !p.isBookmarked } : p)))

      toast({
        title: "Bookmark failed",
        description: "Unable to update bookmark. Please try again.",
        variant: "destructive",
        duration: 3000,
      })
      console.error(e)
    }
  }

  const handleViewFullPaper = (paper: PaperResult) => {
    window.open(paper.url, "_blank")
  }

  return (
    <ErrorBoundary>
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
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-blue-500 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-sm">🎯</span>
            </div>
            <h2 className="text-xl font-extrabold bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent">Navigation</h2>
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
            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-purple-500 via-blue-500 to-indigo-500 rounded-2xl flex items-center justify-center shadow-2xl rotate-3 hover:rotate-0 transition-transform duration-500">
              <span className="text-3xl">🔬</span>
            </div>
            <h3 className="text-lg font-bold bg-gradient-to-r from-purple-700 via-blue-700 to-indigo-700 bg-clip-text text-transparent">Research Hub</h3>
            <p className="text-xs text-slate-600 mt-1">Advanced paper discovery</p>
          </div>

          {/* Chat QA Button - Enhanced */}
          <Button asChild className="w-full justify-start gap-4 h-14 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:from-indigo-600 hover:via-purple-600 hover:to-pink-600 text-white shadow-2xl hover:shadow-indigo-300/50 transition-all duration-500 rounded-2xl font-bold text-base group overflow-hidden relative">
            <Link href="/qa">
              <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/10 to-white/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
              <div className="p-2 bg-white/20 rounded-xl">
                <MessageSquare className="h-6 w-6 group-hover:scale-110 transition-transform duration-300" />
              </div>
              <span className="group-hover:translate-x-1 transition-transform duration-300">💬 Chat QA</span>
            </Link>
          </Button>

          {/* Pagination Mode Toggle - Enhanced */}
          <div className="bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50 border-2 border-emerald-200 rounded-2xl p-4 shadow-xl hover:shadow-emerald-200/50 transition-all duration-300">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-sm">📋</span>
              </div>
              <span className="text-base font-bold bg-gradient-to-r from-emerald-700 to-teal-700 bg-clip-text text-transparent">View Mode</span>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPaginationMode(prev => prev === 'pagination' ? 'load-more' : 'pagination')}
              className={cn(
                "w-full text-sm font-bold transition-all duration-300 rounded-xl border-2 h-10",
                paginationMode === 'pagination'
                  ? "bg-gradient-to-r from-emerald-500 to-teal-500 text-white border-emerald-400 hover:from-emerald-600 hover:to-teal-600 shadow-lg"
                  : "bg-gradient-to-r from-orange-500 to-amber-500 text-white border-orange-400 hover:from-orange-600 hover:to-amber-600 shadow-lg"
              )}
            >
              {paginationMode === 'pagination' ? '📄 Pagination Mode' : '🔄 Load More Mode'}
            </Button>
          </div>

          {/* AI Agent Toggle - Enhanced */}
          <div className={cn(
            "relative overflow-hidden rounded-3xl p-6 shadow-2xl transition-all duration-500 hover:scale-105 cursor-pointer",
            agentMode
              ? "bg-gradient-to-br from-purple-500/20 via-blue-500/20 to-indigo-500/20 border-2 border-purple-300 shadow-purple-200/50"
              : "bg-gradient-to-br from-slate-100 via-white to-slate-50 border-2 border-slate-200 hover:border-purple-200"
          )}
          onClick={() => setAgentMode(!agentMode)}
          >
            {/* Animated background particles */}
            {agentMode && (
              <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -top-2 -left-2 w-4 h-4 bg-purple-400 rounded-full animate-ping opacity-20"></div>
                <div className="absolute top-4 right-4 w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                <div className="absolute bottom-2 left-6 w-3 h-3 bg-indigo-400 rounded-full animate-bounce opacity-30"></div>
              </div>
            )}

            <div className="relative flex items-center gap-4 mb-4">
              <div className={cn(
                "p-3 rounded-2xl transition-all duration-500 shadow-lg",
                agentMode
                  ? "bg-gradient-to-br from-purple-500 via-blue-500 to-indigo-500 shadow-purple-300/50 scale-110"
                  : "bg-gradient-to-br from-slate-200 to-slate-300 hover:from-purple-100 hover:to-blue-100"
              )}>
                <Bot className={cn(
                  "h-6 w-6 transition-all duration-500",
                  agentMode ? "text-white animate-pulse" : "text-slate-600"
                )} />
              </div>
              <div className="flex-1">
                <span className={cn(
                  "text-lg font-bold transition-all duration-500",
                  agentMode
                    ? "bg-gradient-to-r from-purple-700 via-blue-700 to-indigo-700 bg-clip-text text-transparent"
                    : "text-slate-700"
                )}>
                  🤖 AI Agent
                </span>
                <div className={cn(
                  "text-xs transition-all duration-300",
                  agentMode ? "text-purple-600 font-medium" : "text-slate-500"
                )}>
                  {agentMode ? "✨ Active" : "Inactive"}
                </div>
              </div>
            </div>

            <Switch
              checked={agentMode}
              onCheckedChange={setAgentMode}
              className={cn(
                "data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-purple-500 data-[state=checked]:to-blue-500 scale-150 mb-4 transition-all duration-300",
                agentMode && "shadow-lg shadow-purple-300/50"
              )}
            />

            {agentMode && (
              <div className="bg-gradient-to-r from-purple-100/50 via-blue-100/50 to-indigo-100/50 rounded-xl p-3 border border-purple-200/50 backdrop-blur-sm">
                <p className="text-xs text-slate-700 leading-relaxed font-medium">
                  🧠 AI agent will analyze your query and perform intelligent searches with advanced reasoning
                </p>
              </div>
            )}
          </div>

          {/* Stats/Info Section */}
          <div className="bg-gradient-to-br from-amber-50 via-orange-50 to-red-50 border-2 border-amber-200 rounded-2xl p-4 shadow-xl">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-gradient-to-br from-amber-500 to-orange-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-sm">📊</span>
              </div>
              <span className="text-base font-bold bg-gradient-to-r from-amber-700 to-orange-700 bg-clip-text text-transparent">Session Info</span>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Papers Found:</span>
                <span className="font-bold text-amber-700">{searchResults.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Selected:</span>
                <span className="font-bold text-amber-700">{selectedPapers.size}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-600">AI Mode:</span>
                <span className={cn("font-bold", agentMode ? "text-purple-600" : "text-slate-500")}>
                  {agentMode ? "Active ✨" : "Inactive"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col lg:flex-row min-w-0">
        <div className="lg:hidden flex items-center justify-between p-4 border-b border-border bg-card">
          <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(true)}>
            <Menu className="h-4 w-4" />
          </Button>
          <h1 className="font-semibold text-foreground">Paper Search</h1>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPaginationMode(prev => prev === 'pagination' ? 'load-more' : 'pagination')}
            className="text-xs"
          >
            {paginationMode === 'pagination' ? 'Pages' : 'Load More'}
          </Button>
        </div>

        <div className="flex-1 min-w-0">
          <div className="space-y-4">
            <div className="p-4">
              <OfflineIndicator />
            </div>
            <MainSearchInterface
              mode="paper-finder"
              forceMode="paper-finder"
              hideModeSelector
              hideQuickActions
              centerOnEmpty
              fancyHero
              showAgentMode={true}
              onModeChange={() => {}}
              onSearch={(query, filters, qaMode) => handleSearch(query, filters, qaMode, agentMode)}
              isLoading={isLoading}
              selectedPapers={selectedPapers}
              totalPapers={searchResults.length}
            />
          </div>
        </div>

        <div className="w-full lg:w-96 border-t lg:border-t-0 lg:border-l border-border">
          <PaperResultsPanel
            papers={searchResults}
            isLoading={isLoading}
            onPaperSelect={handlePaperSelect}
            onBookmarkToggle={handleBookmarkToggle}
            onViewFullPaper={handleViewFullPaper}
            searchQuery={lastSearchQuery}
            mode="paper-finder"
            selectedPapers={selectedPapers}
            onPaperSelectionChange={handlePaperSelectionChange}
            totalCount={totalCount}
            currentPage={currentPage}
            pageSize={pageSize}
            onPageChange={handlePageChange}
            onLoadMore={handleLoadMore}
            hasNextPage={currentPage * pageSize < totalCount}
            isLoadingMore={isLoadingMore}
            paginationMode={paginationMode}
            onSuggestionSearch={handleSuggestionSearch}
            onClearFilters={handleClearFilters}
            showDiscoverText={true}
          />
        </div>
      </div>
    </div>
    </ErrorBoundary>
  )
}


