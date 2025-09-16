"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { Search, Filter, Calendar, Users, ExternalLink, Quote } from "lucide-react"
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible"

export interface PaperMetadata {
  id: string
  title: string
  authors: string[]
  abstract: string
  url: string
  publishedDate: string
  citationCount: number
  venue: string
  keywords: string[]
  doi?: string
  pdfUrl?: string
  category: string
}

interface SearchFilters {
  dateRange: string
  minCitations: number
  venue: string[]
  category: string[]
  hasFullText: boolean
}

interface PaperSearchEngineProps {
  onPaperSelect?: (paper: PaperMetadata) => void
}

export function PaperSearchEngine({ onPaperSelect }: PaperSearchEngineProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [isSearching, setIsSearching] = useState(false)
  const [showFilters, setShowFilters] = useState(false)
  const [searchResults, setSearchResults] = useState<PaperMetadata[]>([])
  const [filters, setFilters] = useState<SearchFilters>({
    dateRange: "all",
    minCitations: 0,
    venue: [],
    category: [],
    hasFullText: false,
  })

  // Mock data for demonstration
  const mockResults: PaperMetadata[] = [
    {
      id: "1",
      title: "Attention Is All You Need",
      authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
      abstract:
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
      url: "https://arxiv.org/abs/1706.03762",
      publishedDate: "2017-06-12",
      citationCount: 85000,
      venue: "NIPS 2017",
      keywords: ["attention", "transformer", "neural networks", "sequence modeling"],
      doi: "10.48550/arXiv.1706.03762",
      pdfUrl: "https://arxiv.org/pdf/1706.03762.pdf",
      category: "Machine Learning",
    },
    {
      id: "2",
      title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
      abstract:
        "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
      url: "https://arxiv.org/abs/1810.04805",
      publishedDate: "2018-10-11",
      citationCount: 65000,
      venue: "NAACL 2019",
      keywords: ["BERT", "language model", "bidirectional", "pre-training"],
      doi: "10.48550/arXiv.1810.04805",
      pdfUrl: "https://arxiv.org/pdf/1810.04805.pdf",
      category: "Natural Language Processing",
    },
  ]

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setIsSearching(true)
    // Simulate API call
    setTimeout(() => {
      setSearchResults(
        mockResults.filter(
          (paper) =>
            paper.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            paper.abstract.toLowerCase().includes(searchQuery.toLowerCase()) ||
            paper.keywords.some((keyword) => keyword.toLowerCase().includes(searchQuery.toLowerCase())),
        ),
      )
      setIsSearching(false)
    }, 1000)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch()
    }
  }

  const filteredResults = searchResults.filter((paper) => {
    if (filters.minCitations > 0 && paper.citationCount < filters.minCitations) return false
    if (filters.venue.length > 0 && !filters.venue.includes(paper.venue)) return false
    if (filters.category.length > 0 && !filters.category.includes(paper.category)) return false
    if (filters.hasFullText && !paper.pdfUrl) return false

    if (filters.dateRange !== "all") {
      const paperYear = new Date(paper.publishedDate).getFullYear()
      const currentYear = new Date().getFullYear()

      switch (filters.dateRange) {
        case "1year":
          if (currentYear - paperYear > 1) return false
          break
        case "5years":
          if (currentYear - paperYear > 5) return false
          break
        case "10years":
          if (currentYear - paperYear > 10) return false
          break
      }
    }

    return true
  })

  return (
    <div className="flex flex-col h-full">
      {/* Search Header */}
      <div className="p-4 border-b border-border bg-card">
        <div className="flex gap-2 mb-3">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input
              placeholder="Search papers by title, abstract, or keywords..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              className="pl-10"
            />
          </div>
          <Button onClick={handleSearch} disabled={isSearching}>
            {isSearching ? "Searching..." : "Search"}
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => setShowFilters(!showFilters)}>
            <Filter className="h-4 w-4 mr-2" />
            Filters
          </Button>
          {filteredResults.length > 0 && (
            <span className="text-sm text-muted-foreground">{filteredResults.length} results found</span>
          )}
        </div>
      </div>

      {/* Filters Panel */}
      <Collapsible open={showFilters} onOpenChange={setShowFilters}>
        <CollapsibleContent>
          <div className="p-4 border-b border-border bg-muted/50">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Date Range</label>
                <Select
                  value={filters.dateRange}
                  onValueChange={(value) => setFilters((prev) => ({ ...prev, dateRange: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All time</SelectItem>
                    <SelectItem value="1year">Last year</SelectItem>
                    <SelectItem value="5years">Last 5 years</SelectItem>
                    <SelectItem value="10years">Last 10 years</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Min Citations</label>
                <Input
                  type="number"
                  placeholder="0"
                  value={filters.minCitations}
                  onChange={(e) =>
                    setFilters((prev) => ({ ...prev, minCitations: Number.parseInt(e.target.value) || 0 }))
                  }
                />
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Category</label>
                <Select onValueChange={(value) => setFilters((prev) => ({ ...prev, category: [value] }))}>
                  <SelectTrigger>
                    <SelectValue placeholder="All categories" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Machine Learning">Machine Learning</SelectItem>
                    <SelectItem value="Natural Language Processing">NLP</SelectItem>
                    <SelectItem value="Computer Vision">Computer Vision</SelectItem>
                    <SelectItem value="Robotics">Robotics</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="fulltext"
                  checked={filters.hasFullText}
                  onCheckedChange={(checked) => setFilters((prev) => ({ ...prev, hasFullText: checked as boolean }))}
                />
                <label htmlFor="fulltext" className="text-sm font-medium">
                  Full text available
                </label>
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Search Results */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {filteredResults.length === 0 && searchQuery && !isSearching && (
          <div className="text-center py-8 text-muted-foreground">No papers found matching your search criteria.</div>
        )}

        {filteredResults.map((paper) => (
          <Card
            key={paper.id}
            className="hover:shadow-md transition-shadow cursor-pointer"
            onClick={() => onPaperSelect?.(paper)}
          >
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between gap-4">
                <CardTitle className="text-lg leading-tight text-balance">{paper.title}</CardTitle>
                <Badge variant="secondary" className="shrink-0">
                  {paper.category}
                </Badge>
              </div>

              <div className="flex items-center gap-4 text-sm text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Users className="h-4 w-4" />
                  <span>
                    {paper.authors.slice(0, 3).join(", ")}
                    {paper.authors.length > 3 ? " et al." : ""}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <Calendar className="h-4 w-4" />
                  <span>{new Date(paper.publishedDate).getFullYear()}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Quote className="h-4 w-4" />
                  <span>{paper.citationCount.toLocaleString()} citations</span>
                </div>
              </div>
            </CardHeader>

            <CardContent>
              <p className="text-sm text-muted-foreground mb-3 line-clamp-3">{paper.abstract}</p>

              <div className="flex items-center justify-between">
                <div className="flex flex-wrap gap-1">
                  {paper.keywords.slice(0, 4).map((keyword) => (
                    <Badge key={keyword} variant="outline" className="text-xs">
                      {keyword}
                    </Badge>
                  ))}
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">{paper.venue}</span>
                  <Button variant="ghost" size="sm" asChild>
                    <a href={paper.url} target="_blank" rel="noopener noreferrer">
                      <ExternalLink className="h-4 w-4" />
                    </a>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
