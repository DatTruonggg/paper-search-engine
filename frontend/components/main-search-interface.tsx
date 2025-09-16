"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import {
  Search,
  MessageSquare,
  FileText,
  Send,
  Loader2,
  Filter,
  Calendar,
  BookOpen,
  CheckSquare,
  Users,
} from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"

export type SearchMode = "paper-finder" | "qa" | "summary"
export type QAMode = "single-paper" | "all-papers"

interface MainSearchInterfaceProps {
  mode: SearchMode
  onModeChange: (mode: SearchMode) => void
  onSearch: (query: string, filters?: SearchFilters, qaMode?: QAMode) => void
  isLoading?: boolean
  placeholder?: string
  selectedPapers?: Set<string>
  totalPapers?: number
}

export interface SearchFilters {
  dateRange?: {
    from: string
    to: string
  }
  authors?: string[]
  venues?: string[]
  minCitations?: number
  keywords?: string[]
}

export function MainSearchInterface({
  mode,
  onModeChange,
  onSearch,
  isLoading = false,
  placeholder,
  selectedPapers = new Set(),
  totalPapers = 0,
}: MainSearchInterfaceProps) {
  const [query, setQuery] = useState("")
  const [filters, setFilters] = useState<SearchFilters>({})
  const [showFilters, setShowFilters] = useState(false)
  const [qaMode, setQaMode] = useState<QAMode>("single-paper")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSearch(query.trim(), filters, mode === "qa" ? qaMode : undefined)
    }
  }

  const getModeConfig = (searchMode: SearchMode) => {
    switch (searchMode) {
      case "paper-finder":
        return {
          icon: <Search className="h-4 w-4" />,
          label: "Paper Finder",
          description: "Search for academic papers",
          placeholder: "Search for papers on machine learning, neural networks, etc.",
          buttonText: "Find Papers",
        }
      case "qa":
        return {
          icon: <MessageSquare className="h-4 w-4" />,
          label: "Q&A",
          description: "Ask questions about research topics or specific papers",
          placeholder: "Ask a question about your research topic or selected papers...",
          buttonText: "Ask Question",
        }
      case "summary":
        return {
          icon: <FileText className="h-4 w-4" />,
          label: "Summary",
          description: "Get summaries of selected papers or research areas",
          placeholder: "What would you like a summary of? (Select papers from the results panel)",
          buttonText: "Get Summary",
        }
    }
  }

  const currentConfig = getModeConfig(mode)
  const effectivePlaceholder = placeholder || currentConfig.placeholder

  const clearFilters = () => {
    setFilters({})
  }

  const hasActiveFilters = Object.values(filters).some((value) =>
    Array.isArray(value) ? value.length > 0 : value !== undefined,
  )

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Mode Selection */}
      <div className="border-b border-border bg-card">
        <div className="p-4">
          <div className="flex items-center gap-2 mb-4">
            <div className="flex rounded-lg border border-border p-1 bg-muted">
              {(["paper-finder", "qa", "summary"] as const).map((searchMode) => {
                const config = getModeConfig(searchMode)
                return (
                  <Button
                    key={searchMode}
                    variant={mode === searchMode ? "default" : "ghost"}
                    size="sm"
                    onClick={() => onModeChange(searchMode)}
                    className={cn(
                      "flex items-center gap-2 px-3 py-2",
                      mode === searchMode
                        ? "bg-primary text-primary-foreground shadow-sm hover:bg-primary/90"
                        : "text-foreground hover:bg-background/50 hover:text-foreground",
                    )}
                  >
                    {config.icon}
                    {config.label}
                  </Button>
                )
              })}
            </div>
          </div>

          <div className="mb-4">
            <h2 className="text-xl font-semibold text-foreground mb-1">{currentConfig.label}</h2>
            <p className="text-sm text-muted-foreground">{currentConfig.description}</p>
          </div>

          {(mode === "summary" || mode === "qa") && totalPapers > 0 && (
            <div className="mb-4 p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2 text-sm">
                <CheckSquare className="h-4 w-4 text-primary" />
                <span className="font-medium">
                  {selectedPapers.size} of {totalPapers} papers selected
                </span>
              </div>
              {mode === "summary" && selectedPapers.size === 0 && (
                <p className="text-xs text-muted-foreground mt-1">
                  Select papers from the results panel to generate summaries
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Search Interface */}
      <div className="flex-1 flex flex-col">
        <div className="p-6 space-y-4">
          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "qa" && totalPapers > 0 && (
              <div className="p-4 border border-border rounded-lg bg-card">
                <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
                  <Users className="h-4 w-4" />
                  Question Scope
                </h4>
                <RadioGroup value={qaMode} onValueChange={(value: QAMode) => setQaMode(value)} className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="single-paper" id="single-paper" />
                    <Label htmlFor="single-paper" className="text-sm">
                      Ask about selected papers ({selectedPapers.size} selected)
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="all-papers" id="all-papers" />
                    <Label htmlFor="all-papers" className="text-sm">
                      Ask about all papers in results ({totalPapers} papers)
                    </Label>
                  </div>
                </RadioGroup>
              </div>
            )}

            <div className="relative">
              {mode === "qa" || mode === "summary" ? (
                <Textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={effectivePlaceholder}
                  className="min-h-[100px] resize-none pr-12"
                  disabled={isLoading}
                />
              ) : (
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={effectivePlaceholder}
                  className="pr-12"
                  disabled={isLoading}
                />
              )}

              <Button
                type="submit"
                size="sm"
                disabled={
                  !query.trim() ||
                  isLoading ||
                  (mode === "summary" && selectedPapers.size === 0) ||
                  (mode === "qa" && qaMode === "single-paper" && selectedPapers.size === 0)
                }
                className="absolute right-2 top-2"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : mode === "paper-finder" ? (
                  <Search className="h-4 w-4" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>

            {mode === "paper-finder" && (
              <div className="flex items-center gap-2">
                <Popover open={showFilters} onOpenChange={setShowFilters}>
                  <PopoverTrigger asChild>
                    <Button variant="outline" size="sm" className="gap-2 bg-transparent">
                      <Filter className="h-4 w-4" />
                      Filters
                      {hasActiveFilters && (
                        <Badge variant="secondary" className="ml-1 h-4 px-1 text-xs">
                          {
                            Object.values(filters).filter((v) => (Array.isArray(v) ? v.length > 0 : v !== undefined))
                              .length
                          }
                        </Badge>
                      )}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-80 p-4" align="start">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium">Search Filters</h4>
                        {hasActiveFilters && (
                          <Button variant="ghost" size="sm" onClick={clearFilters}>
                            Clear all
                          </Button>
                        )}
                      </div>

                      <div className="space-y-2">
                        <label className="text-sm font-medium flex items-center gap-2">
                          <Calendar className="h-4 w-4" />
                          Publication Year
                        </label>
                        <div className="flex gap-2">
                          <Input
                            placeholder="From"
                            type="number"
                            min="1900"
                            max="2024"
                            value={filters.dateRange?.from || ""}
                            onChange={(e) =>
                              setFilters((prev) => ({
                                ...prev,
                                dateRange: { ...prev.dateRange, from: e.target.value, to: prev.dateRange?.to || "" },
                              }))
                            }
                          />
                          <Input
                            placeholder="To"
                            type="number"
                            min="1900"
                            max="2024"
                            value={filters.dateRange?.to || ""}
                            onChange={(e) =>
                              setFilters((prev) => ({
                                ...prev,
                                dateRange: { ...prev.dateRange, to: e.target.value, from: prev.dateRange?.from || "" },
                              }))
                            }
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <label className="text-sm font-medium flex items-center gap-2">
                          <BookOpen className="h-4 w-4" />
                          Minimum Citations
                        </label>
                        <Select
                          value={filters.minCitations?.toString() || "any"}
                          onValueChange={(value) =>
                            setFilters((prev) => ({
                              ...prev,
                              minCitations: value === "any" ? undefined : Number.parseInt(value),
                            }))
                          }
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Any" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="any">Any</SelectItem>
                            <SelectItem value="10">10+</SelectItem>
                            <SelectItem value="50">50+</SelectItem>
                            <SelectItem value="100">100+</SelectItem>
                            <SelectItem value="500">500+</SelectItem>
                            <SelectItem value="1000">1000+</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </PopoverContent>
                </Popover>

                {hasActiveFilters && (
                  <div className="flex flex-wrap gap-1">
                    {filters.dateRange?.from && (
                      <Badge variant="secondary" className="text-xs">
                        From {filters.dateRange.from}
                      </Badge>
                    )}
                    {filters.dateRange?.to && (
                      <Badge variant="secondary" className="text-xs">
                        To {filters.dateRange.to}
                      </Badge>
                    )}
                    {filters.minCitations && (
                      <Badge variant="secondary" className="text-xs">
                        {filters.minCitations}+ citations
                      </Badge>
                    )}
                  </div>
                )}
              </div>
            )}
          </form>

          <div className="space-y-3">
            <h3 className="text-sm font-medium text-muted-foreground">Quick Actions</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {mode === "paper-finder" && (
                <>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQuery("machine learning transformers")}
                    className="justify-start text-left h-auto p-3"
                  >
                    <div>
                      <div className="font-medium">Transformer Models</div>
                      <div className="text-xs text-muted-foreground">Latest papers on attention mechanisms</div>
                    </div>
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQuery("computer vision deep learning")}
                    className="justify-start text-left h-auto p-3"
                  >
                    <div>
                      <div className="font-medium">Computer Vision</div>
                      <div className="text-xs text-muted-foreground">Deep learning for image processing</div>
                    </div>
                  </Button>
                </>
              )}

              {mode === "qa" && (
                <>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQuery("What are the main contributions of these papers?")}
                    className="justify-start text-left h-auto p-3"
                  >
                    <div>
                      <div className="font-medium">Main Contributions</div>
                      <div className="text-xs text-muted-foreground">Key findings and innovations</div>
                    </div>
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQuery("How do these papers relate to each other?")}
                    className="justify-start text-left h-auto p-3"
                  >
                    <div>
                      <div className="font-medium">Paper Relationships</div>
                      <div className="text-xs text-muted-foreground">Connections and comparisons</div>
                    </div>
                  </Button>
                </>
              )}

              {mode === "summary" && (
                <>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQuery("Provide a comprehensive summary of the selected papers")}
                    className="justify-start text-left h-auto p-3"
                  >
                    <div>
                      <div className="font-medium">Comprehensive Summary</div>
                      <div className="text-xs text-muted-foreground">Detailed overview of selected papers</div>
                    </div>
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQuery("Compare and contrast the methodologies in selected papers")}
                    className="justify-start text-left h-auto p-3"
                  >
                    <div>
                      <div className="font-medium">Methodology Comparison</div>
                      <div className="text-xs text-muted-foreground">Compare approaches and techniques</div>
                    </div>
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
