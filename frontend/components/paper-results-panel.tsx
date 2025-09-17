"use client"

import { useState, memo } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
import {
  ExternalLink,
  Calendar,
  Users,
  Quote,
  BookOpen,
  Search,
  SortAsc,
  SortDesc,
  Eye,
  Download,
  Star,
  StarOff,
  CheckSquare,
  ChevronLeft,
  ChevronRight,
  MoreHorizontal,
  Loader2,
  Copy,
  Award,
  Lock,
  Unlock,
} from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import type { SearchMode } from "./main-search-interface"
import { PaperSkeletonList } from "./paper-skeleton"
import { useToast } from "@/hooks/use-toast"
import { NoResultsFound, SearchWelcome } from "./empty-states"

export interface PaperResult {
  id: string
  title: string
  authors: string[]
  abstract: string
  citationCount: number
  publicationDate: string
  venue?: string
  doi?: string
  url: string
  keywords?: string[]
  pdfUrl?: string
  isBookmarked?: boolean
  isOpenAccess?: boolean
  impactFactor?: number
  journalRank?: string
}

interface PaperResultsPanelProps {
  papers: PaperResult[]
  isLoading?: boolean
  onPaperSelect?: (paper: PaperResult) => void
  onBookmarkToggle?: (paperId: string) => void
  onViewFullPaper?: (paper: PaperResult) => void
  searchQuery?: string
  mode?: SearchMode
  selectedPapers?: Set<string>
  onPaperSelectionChange?: (selectedIds: Set<string>) => void
  // Pagination props
  totalCount?: number
  currentPage?: number
  pageSize?: number
  onPageChange?: (page: number) => void
  onLoadMore?: () => void
  hasNextPage?: boolean
  isLoadingMore?: boolean
  paginationMode?: 'pagination' | 'infinite' | 'load-more'
  // Empty state handlers
  onSuggestionSearch?: (suggestion: string) => void
  onClearFilters?: () => void
  showDiscoverText?: boolean
}

type SortOption = "relevance" | "citations" | "date" | "title"
type SortOrder = "asc" | "desc"

function PaperResultsPanelComponent({
  papers,
  isLoading = false,
  onPaperSelect,
  onBookmarkToggle,
  onViewFullPaper,
  searchQuery,
  mode = "paper-finder",
  selectedPapers = new Set(),
  onPaperSelectionChange,
  totalCount = 0,
  currentPage = 1,
  pageSize = 20,
  onPageChange,
  onLoadMore,
  hasNextPage = false,
  isLoadingMore = false,
  paginationMode = 'pagination',
  onSuggestionSearch,
  onClearFilters,
  showDiscoverText = false,
}: PaperResultsPanelProps) {
  const [searchFilter, setSearchFilter] = useState("")
  const [sortBy, setSortBy] = useState<SortOption>("relevance")
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc")

  const totalPages = Math.ceil(totalCount / pageSize)

  const isSelectionMode = mode === "summary" || mode === "qa"

  // Filter papers based on search
  const filteredPapers = papers.filter((paper) => {
    if (!searchFilter) return true
    const searchLower = searchFilter.toLowerCase()
    return (
      paper.title.toLowerCase().includes(searchLower) ||
      paper.authors.some((author) => author.toLowerCase().includes(searchLower)) ||
      paper.abstract.toLowerCase().includes(searchLower) ||
      paper.venue?.toLowerCase().includes(searchLower)
    )
  })

  // Sort papers
  const sortedPapers = [...filteredPapers].sort((a, b) => {
    let comparison = 0

    switch (sortBy) {
      case "citations":
        comparison = a.citationCount - b.citationCount
        break
      case "date":
        comparison = new Date(a.publicationDate).getTime() - new Date(b.publicationDate).getTime()
        break
      case "title":
        comparison = a.title.localeCompare(b.title)
        break
      case "relevance":
      default:
        // For relevance, we'll use citation count as a proxy
        comparison = a.citationCount - b.citationCount
        break
    }

    return sortOrder === "asc" ? comparison : -comparison
  })

  const handlePaperClick = (paper: PaperResult) => {
    if (isSelectionMode) {
      handlePaperSelection(paper.id)
    } else {
      onPaperSelect?.(paper)
    }
  }

  const handlePaperSelection = (paperId: string) => {
    const newSelection = new Set(selectedPapers)
    if (newSelection.has(paperId)) {
      newSelection.delete(paperId)
    } else {
      newSelection.add(paperId)
    }
    onPaperSelectionChange?.(newSelection)
  }

  const handleSelectAll = () => {
    const allIds = new Set(sortedPapers.map((p) => p.id))
    onPaperSelectionChange?.(allIds)
  }

  const handleSelectNone = () => {
    onPaperSelectionChange?.(new Set())
  }

  const { toast } = useToast()

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      toast({
        title: "Citation copied",
        description: "The citation has been copied to your clipboard.",
        duration: 3000,
      })
    }).catch(() => {
      toast({
        title: "Copy failed",
        description: "Unable to copy citation to clipboard.",
        variant: "destructive",
        duration: 3000,
      })
    })
  }

  const generateCitation = (paper: PaperResult) => {
    // Generate APA style citation
    const authors = paper.authors.join(', ')
    const year = new Date(paper.publicationDate).getFullYear()
    return `${authors} (${year}). ${paper.title}. ${paper.venue || 'Journal'}. ${paper.doi ? `https://doi.org/${paper.doi}` : paper.url}`
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
    })
  }

  const formatCitationCount = (count: number) => {
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}k`
    }
    return count.toString()
  }

  const truncateAbstract = (abstract: string, maxLength = 200) => {
    if (abstract.length <= maxLength) return abstract
    return abstract.substring(0, maxLength) + "..."
  }

  return (
    <div className="flex flex-col h-full bg-card border-l border-border">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-foreground">
            Paper Results
            {papers.length > 0 && (
              <span className="ml-2 text-sm font-normal text-muted-foreground">
                ({filteredPapers.length} of {papers.length})
              </span>
            )}
          </h2>
          {isSelectionMode && (
            <Badge variant="secondary" className="text-xs">
              {mode === "summary" ? "Summary Mode" : "Q&A Mode"}
            </Badge>
          )}
        </div>

        {isSelectionMode && papers.length > 0 && (
          <div className="flex items-center gap-2 mb-3 p-2 bg-muted/50 rounded-lg">
            <CheckSquare className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">{selectedPapers.size} selected</span>
            <div className="flex gap-1 ml-auto">
              <Button variant="ghost" size="sm" onClick={handleSelectAll} className="h-6 px-2 text-xs">
                All
              </Button>
              <Button variant="ghost" size="sm" onClick={handleSelectNone} className="h-6 px-2 text-xs">
                None
              </Button>
            </div>
          </div>
        )}

        {/* Search and Sort Controls */}
        {papers.length > 0 && (
          <div className="space-y-3">
            {/* Search Filter */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Filter papers..."
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                className="pl-9"
              />
            </div>

            {/* Sort Controls */}
            <div className="flex items-center gap-2">
              <Select value={sortBy} onValueChange={(value: SortOption) => setSortBy(value)}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="relevance">Relevance</SelectItem>
                  <SelectItem value="citations">Citations</SelectItem>
                  <SelectItem value="date">Date</SelectItem>
                  <SelectItem value="title">Title</SelectItem>
                </SelectContent>
              </Select>

              <Button
                variant="outline"
                size="sm"
                onClick={() => setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"))}
                className="px-2"
              >
                {sortOrder === "asc" ? <SortAsc className="h-4 w-4" /> : <SortDesc className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      <ScrollArea className="flex-1">
        {isLoading ? (
          <div className="p-2">
            <PaperSkeletonList count={6} />
          </div>
        ) : filteredPapers.length === 0 && papers.length > 0 ? (
          <div className="p-8 text-center">
            <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">No matching papers</h3>
            <p className="text-sm text-muted-foreground mb-4">Try adjusting your search filter or clear it to see all papers</p>
            {onClearFilters && (
              <Button variant="outline" onClick={onClearFilters} className="text-xs">
                Clear filters
              </Button>
            )}
          </div>
        ) : papers.length === 0 && searchQuery ? (
          <NoResultsFound
            searchQuery={searchQuery}
            onSuggestionClick={onSuggestionSearch}
            onClearFilters={onClearFilters}
          />
        ) : papers.length === 0 ? (
          <SearchWelcome
            onSuggestionClick={onSuggestionSearch}
            showDiscoverText={showDiscoverText}
          />
        ) : (
          <div className="p-2 space-y-2">
            {sortedPapers.map((paper) => (
              <Card
                key={paper.id}
                className={cn(
                  "group cursor-pointer transition-all duration-300 hover:shadow-2xl hover:shadow-purple-100 hover:border-purple-300 hover:-translate-y-1 hover:scale-[1.01]",
                  selectedPapers.has(paper.id) && "ring-2 ring-purple-400 bg-gradient-to-r from-purple-50 to-blue-50 shadow-lg",
                  isSelectionMode && "hover:bg-gradient-to-r hover:from-purple-50/50 hover:to-blue-50/50",
                  "border-l-4 border-l-transparent hover:border-l-purple-400 bg-white/90 backdrop-blur-sm border border-slate-200"
                )}
                onClick={() => handlePaperClick(paper)}
              >
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-start gap-2 flex-1">
                      {isSelectionMode && (
                        <Checkbox
                          checked={selectedPapers.has(paper.id)}
                          onCheckedChange={() => handlePaperSelection(paper.id)}
                          onClick={(e) => e.stopPropagation()}
                          className="mt-1"
                        />
                      )}
                      <h3 className="font-medium text-sm leading-tight line-clamp-2 flex-1">{paper.title}</h3>
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 w-6 p-0"
                              onClick={(e) => {
                                e.stopPropagation()
                                onBookmarkToggle?.(paper.id)
                              }}
                            >
                              {paper.isBookmarked ? (
                                <Star className="h-3 w-3 fill-current text-yellow-500" />
                              ) : (
                                <StarOff className="h-3 w-3" />
                              )}
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>{paper.isBookmarked ? "Remove bookmark" : "Add bookmark"}</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  </div>

                  {/* Authors */}
                  <div
                    className={cn("flex items-center gap-1 text-xs text-muted-foreground", isSelectionMode && "ml-6")}
                  >
                    <Users className="h-3 w-3" />
                    <span className="line-clamp-1">
                      {paper.authors.slice(0, 3).join(", ")}
                      {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
                    </span>
                  </div>
                </CardHeader>

                <CardContent className="pt-0">
                  <div className={cn(isSelectionMode && "ml-6")}>
                    {/* Metadata */}
                    <div className="flex items-center gap-4 mb-3 text-xs text-muted-foreground flex-wrap">
                      <div className="flex items-center gap-1">
                        <Quote className="h-3 w-3" />
                        <span className={cn(
                          paper.citationCount > 100 && "font-bold bg-gradient-to-r from-amber-600 to-orange-600 bg-clip-text text-transparent",
                          paper.citationCount <= 100 && "text-slate-600 font-medium"
                        )}>
                          {formatCitationCount(paper.citationCount)} citations
                        </span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        <span>{formatDate(paper.publicationDate)}</span>
                      </div>
                      {paper.impactFactor && (
                        <div className="flex items-center gap-1">
                          <Award className="h-3 w-3" />
                          <span>IF: {paper.impactFactor}</span>
                        </div>
                      )}
                      {paper.isOpenAccess && (
                        <Badge className="text-xs px-2 py-1 bg-gradient-to-r from-emerald-100 to-green-100 text-emerald-700 border border-emerald-200 shadow-sm font-medium">
                          <Unlock className="h-2 w-2 mr-1" />
                          Open Access
                        </Badge>
                      )}
                    </div>

                    {/* Venue */}
                    {paper.venue && (
                      <div className="mb-3">
                        <Badge className="text-xs bg-gradient-to-r from-slate-100 to-slate-200 text-slate-700 border border-slate-300 shadow-sm font-medium">
                          {paper.venue}
                          {paper.journalRank && (
                            <span className="ml-1 text-xs bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent font-bold">({paper.journalRank})</span>
                          )}
                        </Badge>
                      </div>
                    )}

                    {/* Abstract */}
                    <p className="text-xs text-muted-foreground leading-relaxed mb-3 line-clamp-3">
                      {truncateAbstract(paper.abstract)}
                    </p>

                    {/* Keywords */}
                    {paper.keywords && paper.keywords.length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-3">
                        {paper.keywords.slice(0, 3).map((keyword, index) => (
                          <Badge key={index} variant="outline" className="text-xs px-1 py-0">
                            {keyword}
                          </Badge>
                        ))}
                        {paper.keywords.length > 3 && (
                          <Badge variant="outline" className="text-xs px-1 py-0">
                            +{paper.keywords.length - 3}
                          </Badge>
                        )}
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex-1 h-8 text-xs bg-gradient-to-r from-slate-50 to-white hover:from-purple-50 hover:to-blue-50 border border-slate-200 hover:border-purple-300 transition-all duration-300 shadow-sm hover:shadow-md font-semibold text-slate-700 hover:text-purple-700"
                        onClick={(e) => {
                          e.stopPropagation()
                          onViewFullPaper?.(paper)
                        }}
                      >
                        <Eye className="h-3 w-3 mr-1" />
                        View Paper
                      </Button>

                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8 w-8 p-0 text-xs bg-gradient-to-r from-slate-50 to-white hover:from-purple-50 hover:to-blue-50 border border-slate-200 hover:border-purple-300 transition-all duration-300 shadow-sm hover:shadow-md"
                        onClick={(e) => {
                          e.stopPropagation()
                          copyToClipboard(generateCitation(paper))
                        }}
                      >
                        <Copy className="h-3 w-3 text-slate-600 hover:text-purple-600 transition-colors" />
                      </Button>

                      {paper.pdfUrl && (
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-7 w-7 p-0 bg-transparent"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  window.open(paper.pdfUrl, "_blank")
                                }}
                              >
                                <Download className="h-3 w-3" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>Download PDF</TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      )}

                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-7 w-7 p-0 bg-transparent"
                              onClick={(e) => {
                                e.stopPropagation()
                                window.open(paper.url, "_blank")
                              }}
                            >
                              <ExternalLink className="h-3 w-3" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Open in new tab</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}

            {/* Load More Button */}
            {paginationMode === 'load-more' && hasNextPage && (
              <div className="p-4 text-center">
                <Button
                  onClick={onLoadMore}
                  disabled={isLoadingMore}
                  variant="outline"
                  className="w-full"
                >
                  {isLoadingMore ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      Loading more papers...
                    </>
                  ) : (
                    <>Load More ({totalCount - papers.length} remaining)</>
                  )}
                </Button>
              </div>
            )}
          </div>
        )}
      </ScrollArea>

      {/* Pagination Controls */}
      {paginationMode === 'pagination' && totalPages > 1 && papers.length > 0 && (
        <div className="p-3 border-t border-border bg-card">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              Showing {((currentPage - 1) * pageSize) + 1}-{Math.min(currentPage * pageSize, totalCount)} of {totalCount} papers
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => onPageChange?.(currentPage - 1)}
                disabled={currentPage === 1 || isLoading}
                className="h-8 w-8 p-0"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>

              {/* Page numbers */}
              <div className="flex items-center gap-1">
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  let pageNum: number
                  if (totalPages <= 5) {
                    pageNum = i + 1
                  } else if (currentPage <= 3) {
                    pageNum = i + 1
                  } else if (currentPage >= totalPages - 2) {
                    pageNum = totalPages - 4 + i
                  } else {
                    pageNum = currentPage - 2 + i
                  }

                  return (
                    <Button
                      key={pageNum}
                      variant={currentPage === pageNum ? "default" : "ghost"}
                      size="sm"
                      onClick={() => onPageChange?.(pageNum)}
                      disabled={isLoading}
                      className="h-8 w-8 p-0 text-xs"
                    >
                      {pageNum}
                    </Button>
                  )
                })}

                {totalPages > 5 && currentPage < totalPages - 2 && (
                  <>
                    <MoreHorizontal className="h-4 w-4 text-muted-foreground" />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onPageChange?.(totalPages)}
                      disabled={isLoading}
                      className="h-8 w-8 p-0 text-xs"
                    >
                      {totalPages}
                    </Button>
                  </>
                )}
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={() => onPageChange?.(currentPage + 1)}
                disabled={currentPage === totalPages || isLoading}
                className="h-8 w-8 p-0"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Selection Status - only show if not using pagination */}
      {isSelectionMode && selectedPapers.size > 0 && paginationMode !== 'pagination' && (
        <div className="p-3 border-t border-border bg-primary/5">
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium text-primary">
              {selectedPapers.size} paper{selectedPapers.size !== 1 ? "s" : ""} selected for{" "}
              {mode === "summary" ? "summarization" : "Q&A"}
            </div>
            <Button variant="ghost" size="sm" onClick={handleSelectNone} className="text-xs h-6">
              Clear
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

export const PaperResultsPanel = memo(PaperResultsPanelComponent)
