"use client"

import { useState } from "react"
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
} from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import type { SearchMode } from "./main-search-interface"

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
}

type SortOption = "relevance" | "citations" | "date" | "title"
type SortOrder = "asc" | "desc"

export function PaperResultsPanel({
  papers,
  isLoading = false,
  onPaperSelect,
  onBookmarkToggle,
  onViewFullPaper,
  searchQuery,
  mode = "paper-finder",
  selectedPapers = new Set(),
  onPaperSelectionChange,
}: PaperResultsPanelProps) {
  const [searchFilter, setSearchFilter] = useState("")
  const [sortBy, setSortBy] = useState<SortOption>("relevance")
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc")

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
          <div className="p-4 space-y-4">
            {[...Array(3)].map((_, i) => (
              <Card key={i} className="animate-pulse">
                <CardHeader className="pb-2">
                  <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                  <div className="h-3 bg-muted rounded w-1/2" />
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="h-3 bg-muted rounded w-full" />
                    <div className="h-3 bg-muted rounded w-5/6" />
                    <div className="h-3 bg-muted rounded w-4/6" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : papers.length === 0 ? (
          <div className="p-8 text-center">
            <BookOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">No papers found</h3>
            <p className="text-sm text-muted-foreground">
              {searchQuery ? "Try adjusting your search query or filters" : "Start searching to see paper results here"}
            </p>
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {sortedPapers.map((paper) => (
              <Card
                key={paper.id}
                className={cn(
                  "cursor-pointer transition-all hover:shadow-md",
                  selectedPapers.has(paper.id) && "ring-2 ring-primary",
                  isSelectionMode && "hover:bg-muted/50",
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
                    <div className="flex items-center gap-4 mb-3 text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Quote className="h-3 w-3" />
                        <span>{formatCitationCount(paper.citationCount)} citations</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        <span>{formatDate(paper.publicationDate)}</span>
                      </div>
                    </div>

                    {/* Venue */}
                    {paper.venue && (
                      <Badge variant="secondary" className="mb-3 text-xs">
                        {paper.venue}
                      </Badge>
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
                        className="flex-1 h-7 text-xs bg-transparent"
                        onClick={(e) => {
                          e.stopPropagation()
                          onViewFullPaper?.(paper)
                        }}
                      >
                        <Eye className="h-3 w-3 mr-1" />
                        View Full Paper
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
          </div>
        )}
      </ScrollArea>

      {isSelectionMode && selectedPapers.size > 0 && (
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
