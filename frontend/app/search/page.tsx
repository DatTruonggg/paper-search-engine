"use client"

import { useEffect, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import {
  MainSearchInterface,
  type SearchFilters,
  type QAMode,
} from "@/components/main-search-interface"
import { PaperResultsPanel, type PaperResult } from "@/components/paper-results-panel"
import { Menu, X, Bot, MessageSquare, Bookmark, Trash2, ExternalLink } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
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
  const [bookmarks, setBookmarks] = useState<PaperResult[]>([])
  const [bookmarksOpen, setBookmarksOpen] = useState(false)
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

  // Load bookmarks from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem("pse:bookmarks")
      if (!raw) return
      const saved: PaperResult[] = JSON.parse(raw)
      setBookmarks(Array.isArray(saved) ? saved : [])
    } catch {}
  }, [])

  const persistBookmarks = (items: PaperResult[]) => {
    try {
      localStorage.setItem("pse:bookmarks", JSON.stringify(items))
    } catch {}
  }

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

  const handleSearch = async (
    query: string,
    filters?: SearchFilters,
    _qaMode?: QAMode,
    isAgentMode?: boolean,
    backendMode: 'fulltext' | 'hybrid' | 'semantic' = 'fulltext',
    page = 1,
  ) => {
    setIsLoading(true)
    setLastSearchQuery(query)
    if (page === 1) {
      setCurrentPage(1)
      setSearchResults([])
      setSelectedPapers(new Set())
    }
    try {
      // Use agent search if enabled
      console.log('[search] POST', { query, filters, isAgentMode, mode: backendMode, page })
      try {
        // reset stepIndex to 0 on new agent searches
        localStorage.setItem('pse:searchProgress', JSON.stringify({ inProgress: true, isAgent: !!isAgentMode, stepIndex: 0, start: Date.now(), mode: backendMode }))
      } catch {}
      const resp = isAgentMode
        ? await api.searchPapersWithAgent(query, filters, page, pageSize)
        : await api.searchPapers(query, filters, page, pageSize, backendMode)
      console.log('[search] RESPONSE', resp)

      // Merge and de-duplicate results by id in a single pass, and persist the exact list we render
      let nextPapers: PaperResult[] = []
      setSearchResults(prev => {
        const base = page === 1 ? [] : prev
        const merged = [...base, ...resp.papers]
        const seen = new Set<string>()
        const dedup: PaperResult[] = []
        for (const p of merged) {
          if (!seen.has(p.id)) {
            seen.add(p.id)
            dedup.push(p)
          }
        }
        nextPapers = dedup
        return dedup
      })

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

      persist({ contextId: resp.contextId, papers: nextPapers, selectedIds: [] })
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
      try {
        localStorage.setItem('pse:searchProgress', JSON.stringify({ inProgress: false }))
      } catch {}
    }
  }

  const handlePageChange = (page: number) => {
    console.log('[pagination] change', { page })
    // Non-agent pagination only; agent returns a single page
    handleSearch(lastSearchQuery, undefined, undefined, false, 'fulltext', page)
  }

  const handleLoadMore = async () => {
    if (isLoadingMore || isLoading) return
    setIsLoadingMore(true)
    console.log('[pagination] loadMore', { nextPage: currentPage + 1 })
    await handleSearch(lastSearchQuery, undefined, undefined, false, 'fulltext', currentPage + 1)
  }

  const handleSuggestionSearch = (suggestion: string) => {
    console.log('[suggestion] select', suggestion)
    handleSearch(suggestion, undefined, undefined, false, 'fulltext', 1)
  }

  const handleClearFilters = () => {
    console.log('[filters] clear')
    // This would be implemented to clear any active filters
    // For now, we'll just trigger a new search with the current query
    if (lastSearchQuery) {
      handleSearch(lastSearchQuery, undefined, undefined, false, 'fulltext', 1)
    }
  }

  const handlePaperSelect = (paper: PaperResult) => {
    console.log('[paper] select', { id: paper.id, title: paper.title })
  }

  const handlePaperSelectionChange = (newSelection: Set<string>) => {
    console.log('[paper] selectionChange', Array.from(newSelection))
    setSelectedPapers(newSelection)
    persist({ selectedIds: Array.from(newSelection) })
  }

  const handleBookmarkToggle = async (paperId: string) => {
    console.log('[bookmark] toggle', paperId)
    const paper = searchResults.find((p) => p.id === paperId)
    const wasBookmarked = paper?.isBookmarked

    // Optimistically update UI
    setSearchResults((prev) => prev.map((p) => (p.id === paperId ? { ...p, isBookmarked: !p.isBookmarked } : p)))

    try {
      const on = !wasBookmarked
      await api.bookmark(paperId, on ?? true)

      // Persist to localStorage bookmarks
      if (paper) {
        console.log('[bookmark] persist', { add: on })
        if (on) {
          const exists = bookmarks.some(b => b.id === paper.id)
          const next = exists ? bookmarks : [...bookmarks, paper]
          setBookmarks(next)
          persistBookmarks(next)
        } else {
          const next = bookmarks.filter(b => b.id !== paper.id)
          setBookmarks(next)
          persistBookmarks(next)
        }
      }

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
    console.log('[paper] open', paper.id)
    window.open(paper.url, "_blank")
  }

  // Save all current results to bookmarks (dedupe by id)
  // Removed per new spec: save via star on each paper

  const handleDeleteBookmark = (paperId: string) => {
    console.log('[bookmark] delete', paperId)
    const next = bookmarks.filter(b => b.id !== paperId)
    setBookmarks(next)
    persistBookmarks(next)
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
        bg-gradient-to-br from-green-50 via-white to-emerald-50
        border-r-2 border-green-200
        shadow-xl shadow-green-100/30
        transform transition-all duration-300 ease-in-out lg:transform-none
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
      `}
      >
        {/* Mobile Header */}
        <div className="flex items-center justify-between p-6 border-b border-green-200 lg:hidden bg-gradient-to-r from-green-50 to-emerald-50 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-green-600 to-emerald-600 rounded-xl flex items-center justify-center shadow-md">
              <span className="text-white font-bold text-sm">🍀</span>
            </div>
            <h2 className="text-xl font-bold text-green-800">Navigation</h2>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(false)}
            className="hover:bg-green-100 transition-colors duration-200 rounded-lg border border-green-200"
          >
            <X className="h-5 w-5 text-green-600" />
          </Button>
        </div>

        {/* Sidebar Content */}
        <div className="p-6 space-y-6 h-full overflow-y-auto">
          {/* Welcome Section */}
          <div className="text-center pb-4 border-b border-green-100">
            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-green-600 to-emerald-600 rounded-2xl flex items-center justify-center shadow-lg hover:shadow-xl transition-all duration-300">
              <span className="text-3xl">🔬</span>
            </div>
            <h3 className="text-lg font-bold text-green-800">Research Hub</h3>
            <p className="text-xs text-slate-600 mt-1">Advanced paper discovery</p>
          </div>

          {/* Chat QA Button - Enhanced */}
          <Button asChild className="w-full justify-start gap-4 h-14 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white shadow-lg hover:shadow-xl transition-all duration-300 rounded-xl font-semibold text-base group overflow-hidden relative">
            <Link href="/qa">
              <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/10 to-white/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
              <div className="p-2 bg-white/20 rounded-lg">
                <MessageSquare className="h-6 w-6 group-hover:scale-105 transition-transform duration-300" />
              </div>
              <span className="group-hover:translate-x-1 transition-transform duration-300">Chat QA</span>
            </Link>
          </Button>

          {/* Bookmarks Section */}
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200 rounded-lg p-4 shadow-sm hover:shadow-md transition-all duration-200">
            <button
              className="w-full flex items-center justify-between p-2 rounded-md hover:bg-green-100 transition-colors"
              onClick={() => setBookmarksOpen(true)}
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-to-br from-green-600 to-emerald-600 rounded-lg flex items-center justify-center shadow-sm">
                  <Bookmark className="h-4 w-4 text-white" />
                </div>
                <span className="text-base font-semibold text-green-800">Bookmarks</span>
              </div>
              <span className="text-xs text-slate-600">{bookmarks.length} saved</span>
            </button>
          </div>

          {/* AI Agent Toggle - Enhanced */}
          <div className={cn(
            "relative overflow-hidden rounded-lg p-5 shadow-sm transition-all duration-200 hover:shadow-md cursor-pointer",
            agentMode
              ? "bg-gradient-to-br from-green-100 to-emerald-100 border border-green-300"
              : "bg-white border border-slate-200 hover:border-green-200"
          )}
          onClick={() => setAgentMode(!agentMode)}
          >
            {/* Animated background particles */}
            {agentMode && (
              <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -top-2 -left-2 w-4 h-4 bg-green-400 rounded-full animate-ping opacity-20"></div>
                <div className="absolute top-4 right-4 w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                <div className="absolute bottom-2 left-6 w-3 h-3 bg-teal-400 rounded-full animate-bounce opacity-30"></div>
              </div>
            )}

            <div className="relative flex items-center gap-4 mb-4">
              <div className={cn(
                "p-3 rounded-lg transition-all duration-300 shadow-sm",
                agentMode
                  ? "bg-gradient-to-br from-green-600 to-emerald-600 shadow-green-300/30 scale-105"
                  : "bg-gradient-to-br from-slate-100 to-slate-200 hover:from-green-100 hover:to-emerald-100"
              )}>
                <Bot className={cn(
                  "h-6 w-6 transition-all duration-300",
                  agentMode ? "text-white" : "text-slate-600"
                )} />
              </div>
              <div className="flex-1">
                <span className={cn(
                  "text-lg font-semibold transition-all duration-300",
                  agentMode
                    ? "text-green-800"
                    : "text-slate-700"
                )}>
                  AI Agent
                </span>
                <div className={cn(
                  "text-xs transition-all duration-200",
                  agentMode ? "text-green-600 font-medium" : "text-slate-500"
                )}>
                  {agentMode ? "Active" : "Inactive"}
                  <p className="text-xs text-slate-500">
                    {agentMode ? "AI agent will analyze your query and perform intelligent searches with advanced reasoning." : "AI agent is inactive."}
                  </p>
                </div>
              </div>
            </div>

            <Switch
              checked={agentMode}
              onCheckedChange={setAgentMode}
              className={cn(
                "data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-green-600 data-[state=checked]:to-emerald-600 scale-125 mb-4 transition-all duration-200",
                agentMode && "shadow-sm shadow-green-300/30"
              )}
            />

            {agentMode && (
              <div className="bg-red-50 rounded-lg p-3 border border-green-200">
                <p className="text-xs text-slate-700 leading-relaxed">
                  It may take a while to complete.
                </p>
              </div>
            )}
          </div>

          {/* Stats/Info Section */}
          <div className="bg-gradient-to-br from-emerald-50 to-teal-50 border border-emerald-200 rounded-lg p-4 shadow-sm">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-gradient-to-br from-emerald-600 to-teal-600 rounded-lg flex items-center justify-center shadow-sm">
                <span className="text-white text-sm">📊</span>
              </div>
              <span className="text-base font-semibold text-emerald-800">Session Info</span>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Papers Found:</span>
                <span className="font-bold text-emerald-700">{searchResults.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Selected:</span>
                <span className="font-bold text-emerald-700">{selectedPapers.size}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-600">AI Mode:</span>
                <span className={cn("font-bold", agentMode ? "text-green-600" : "text-slate-500")}>
                  {agentMode ? "Active" : "Inactive"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bookmarks Dialog */}
      <Dialog open={bookmarksOpen} onOpenChange={setBookmarksOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Saved bookmarks</DialogTitle>
          </DialogHeader>
          {bookmarks.length === 0 ? (
            <p className="text-sm text-slate-600">You have no saved papers yet. Click the star on a paper to add it.</p>
          ) : (
            <div className="space-y-2 max-h-96 overflow-auto pr-1">
              {bookmarks.map((b) => (
                <div key={b.id} className="group flex items-start gap-2 p-2 rounded-md border border-slate-200 bg-white hover:bg-slate-50 transition-colors">
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-slate-800 line-clamp-2">{b.title}</div>
                    <div className="text-xs text-slate-500 line-clamp-1">{b.authors.slice(0,3).join(', ')}{b.authors.length>3?` +${b.authors.length-3}`:''}</div>
                  </div>
                  <div className="flex gap-1 flex-shrink-0">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 px-2"
                      onClick={() => window.open(b.pdfUrl || b.url, '_blank')}
                    >
                      <ExternalLink className="h-3 w-3 mr-1" /> Open
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 w-7 p-0 text-red-600 hover:text-red-700"
                      onClick={() => handleDeleteBookmark(b.id)}
                      title="Delete"
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </DialogContent>
      </Dialog>

      <div className="flex-1 flex flex-col lg:flex-row min-w-0 min-h-0">
        <div className="lg:hidden flex items-center justify-between p-4 border-b border-slate-200 bg-white">
          <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(true)}>
            <Menu className="h-4 w-4" />
          </Button>
          <h1 className="font-semibold text-slate-800">Paper Search</h1>
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
              forceAgentMode={agentMode}
              onModeChange={() => {}}
              onSearch={(query, filters, qaMode) => handleSearch(query, filters, qaMode, agentMode)}
              isLoading={isLoading}
              selectedPapers={selectedPapers}
              totalPapers={searchResults.length}
            />
          </div>
        </div>

        <div className="w-full lg:w-96 border-t lg:border-t-0 lg:border-l border-border h-full overflow-hidden">
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


