"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback, useMemo, memo } from "react"
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
  Bot,
  Zap,
} from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"
import { Switch } from "@/components/ui/switch"
import { Command, CommandGroup, CommandItem, CommandList } from "@/components/ui/command"
import { useSearchShortcuts } from "@/hooks/use-keyboard-shortcuts"
import { useSearchSuggestions } from "@/hooks/use-search-suggestions"
import { useDebounce } from "@/hooks/use-debounce"

export type SearchMode = "paper-finder" | "qa" | "summary"
export type QAMode = "single-paper" | "all-papers"

interface MainSearchInterfaceProps {
  mode: SearchMode
  onModeChange: (mode: SearchMode) => void
  onSearch: (query: string, filters?: SearchFilters, qaMode?: QAMode, isAgentMode?: boolean) => void
  isLoading?: boolean
  placeholder?: string
  selectedPapers?: Set<string>
  totalPapers?: number
  /**
   * When true, the mode selector header is hidden. Use with forceMode to lock the UI mode.
   */
  hideModeSelector?: boolean
  /**
   * When provided, the UI behaves as if this mode is selected and disables switching.
   */
  forceMode?: SearchMode
  /**
   * When true, hides the Quick Actions section.
   */
  hideQuickActions?: boolean
  /**
   * When true and there are no results, center the search box like a hero.
   */
  centerOnEmpty?: boolean
  /**
   * Apply enhanced hero styling when centered.
   */
  fancyHero?: boolean
  /**
   * When true, shows the AI agent toggle (only for search page)
   */
  showAgentMode?: boolean
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
  openAccess?: boolean
}

const FILTER_PRESETS = [
  {
    name: "Recent & Highly Cited",
    description: "Papers from last 3 years with 50+ citations",
    filters: {
      dateRange: { from: "2021", to: "2024" },
      minCitations: 50
    }
  },
  {
    name: "Latest Research",
    description: "Papers from the current year",
    filters: {
      dateRange: { from: "2024", to: "2024" }
    }
  },
  {
    name: "Open Access",
    description: "Freely accessible papers",
    filters: {
      openAccess: true
    }
  },
  {
    name: "High Impact",
    description: "Papers with 100+ citations",
    filters: {
      minCitations: 100
    }
  },
  {
    name: "Last Decade",
    description: "Papers from 2014-2024",
    filters: {
      dateRange: { from: "2014", to: "2024" }
    }
  }
]

function MainSearchInterfaceComponent({
  mode,
  onModeChange,
  onSearch,
  isLoading = false,
  placeholder,
  selectedPapers = new Set(),
  totalPapers = 0,
  hideModeSelector = false,
  forceMode,
  hideQuickActions = false,
  centerOnEmpty = false,
  fancyHero = false,
  showAgentMode = false,
}: MainSearchInterfaceProps) {
  const [query, setQuery] = useState("")
  const [filters, setFilters] = useState<SearchFilters>({})
  const [showFilters, setShowFilters] = useState(false)
  const [qaMode, setQaMode] = useState<QAMode>("single-paper")
  const [agentMode, setAgentMode] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-save query to localStorage
  useEffect(() => {
    const saved = localStorage.getItem('pse:lastQuery')
    if (saved && !query) {
      setQuery(saved)
    }
  }, [])

  useEffect(() => {
    if (query) {
      localStorage.setItem('pse:lastQuery', query)
    }
  }, [query])

  // Debounce query for suggestions
  const debouncedQuery = useDebounce(query, 300)

  // Search suggestions
  const { suggestions } = useSearchSuggestions(debouncedQuery, showSuggestions)

  // Keyboard shortcuts
  const focusSearchInput = () => {
    if (searchInputRef.current) {
      searchInputRef.current.focus()
    } else if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  const clearSearch = () => {
    setQuery('')
    setShowSuggestions(false)
    focusSearchInput()
  }

  useSearchShortcuts(focusSearchInput, clearSearch)

  // Determine the effective mode taking into account forcing from parent
  const effectiveMode: SearchMode = forceMode ?? mode

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSearch(query.trim(), filters, effectiveMode === "qa" ? qaMode : undefined, agentMode)
    }
  }, [query, filters, effectiveMode, qaMode, agentMode, onSearch])

  const getModeConfig = (searchMode: SearchMode) => {
    switch (searchMode) {
      case "paper-finder":
        return {
          icon: <Search className="h-4 w-4" />,
          label: "🔍 Research Discovery Engine",
          description: "✨ Discover groundbreaking academic papers with intelligent search",
          placeholder: "Search for papers on machine learning, neural networks, etc.",
          buttonText: "🚀 Discover Papers",
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

  const currentConfig = useMemo(() => getModeConfig(effectiveMode), [effectiveMode])
  const effectivePlaceholder = placeholder || currentConfig.placeholder

  const clearFilters = useCallback(() => {
    setFilters({})
  }, [])

  const hasActiveFilters = Object.values(filters).some((value) =>
    Array.isArray(value) ? value.length > 0 : value !== undefined && value !== false,
  )

  const applyFilterPreset = useCallback((preset: typeof FILTER_PRESETS[0]) => {
    setFilters(preset.filters)
    setShowFilters(true)
  }, [])

  const isHero = effectiveMode === "paper-finder" && centerOnEmpty && totalPapers === 0

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Mode Selection */}
      <div className="border-b border-border bg-card">
        <div className="p-4">
          {!hideModeSelector && (
            <div className="flex items-center gap-2 mb-4">
              <div className="flex rounded-lg border border-border p-1 bg-muted">
                {(["paper-finder", "qa", "summary"] as const).map((searchMode) => {
                  const config = getModeConfig(searchMode)
                  const isActive = effectiveMode === searchMode
                  return (
                    <Button
                      key={searchMode}
                      variant={isActive ? "default" : "ghost"}
                      size="sm"
                      onClick={() => onModeChange(searchMode)}
                      className={cn(
                        "flex items-center gap-2 px-3 py-2",
                        isActive
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
          )}

          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <div>
                <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent mb-1">{currentConfig.label}</h2>
                <p className="text-sm font-medium bg-gradient-to-r from-slate-600 to-slate-500 bg-clip-text text-transparent">{currentConfig.description}</p>
              </div>

            </div>

            {agentMode && (
              <div className="mb-3 p-4 bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-cyan-500/10 border border-purple-200 rounded-xl backdrop-blur-sm shadow-lg">
                <div className="flex items-center gap-3 text-sm">
                  <div className="p-1.5 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full shadow-md">
                    <Zap className="h-3 w-3 text-white" />
                  </div>
                  <span className="font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                    🤖 AI Agent Mode Active
                  </span>
                </div>
                <p className="text-xs text-slate-600 mt-2 leading-relaxed ml-6">
                  The AI agent will analyze your query and perform intelligent multi-step searches
                </p>
              </div>
            )}
          </div>

          {(effectiveMode === "summary" || effectiveMode === "qa") && totalPapers > 0 && (
            <div className="mb-4 p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2 text-sm">
                <CheckSquare className="h-4 w-4 text-primary" />
                <span className="font-medium">
                  {selectedPapers.size} of {totalPapers} papers selected
                </span>
              </div>
              {effectiveMode === "summary" && selectedPapers.size === 0 && (
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
        {isHero ? (
          <div className={cn("flex-1 flex items-center justify-center relative",
            fancyHero && "bg-gradient-to-b from-transparent to-muted/30")}
          >
            {isLoading && (
              <div className="absolute inset-0 bg-gradient-to-br from-white/95 to-slate-100/95 backdrop-blur-md z-10 flex items-center justify-center">
                <div className="flex flex-col items-center gap-4 p-8 rounded-3xl bg-white/90 shadow-2xl border border-white/60">
                  <div className="relative">
                    <div className="absolute inset-0 animate-ping rounded-full bg-gradient-to-r from-purple-400 to-blue-400 opacity-75"></div>
                    <div className="relative p-3 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full shadow-xl">
                      <Loader2 className="h-6 w-6 animate-spin text-white" />
                    </div>
                  </div>
                  <p className="text-base font-semibold bg-gradient-to-r from-slate-700 to-slate-600 bg-clip-text text-transparent">
                    {agentMode ? "🤖 AI Agent is searching..." : "🔍 Searching papers..."}
                  </p>
                </div>
              </div>
            )}
            <div className="max-w-2xl w-full px-6 py-8">
              <div className="text-center mb-10">
                <h1 className="text-4xl lg:text-6xl font-black bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent mb-4 leading-tight animate-pulse">
                  🔬 Research Discovery Hub
                </h1>
                {showAgentMode && (
                  <p className="text-slate-600 mt-4 text-xl font-semibold">
                    ✨ Discover groundbreaking research with AI-powered search
                  </p>
                )}
                {!showAgentMode && (
                  <p className="text-slate-600 mt-4 text-xl font-semibold">
                    Search through academic papers and research
                  </p>
                )}
              </div>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="relative">
                  <Input
                    ref={searchInputRef}
                    value={query}
                    onChange={(e) => {
                      setQuery(e.target.value)
                      setShowSuggestions(true)
                    }}
                    onFocus={() => setShowSuggestions(true)}
                    onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                    placeholder={effectivePlaceholder}
                    className="pr-12 h-12 lg:h-14 text-base lg:text-lg shadow-sm"
                    disabled={isLoading}
                  />
                  <Button
                    type="submit"
                    size="sm"
                    disabled={!query.trim() || isLoading}
                    className="absolute right-2 top-2 lg:top-2.5 h-8 lg:h-9"
                  >
                    {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                  </Button>

                  {/* Search suggestions dropdown */}
                  {showSuggestions && suggestions.length > 0 && (
                    <div className="absolute top-full left-0 right-0 mt-1 bg-popover border border-border rounded-md shadow-lg z-50">
                      <Command>
                        <CommandList className="max-h-48">
                          <CommandGroup>
                            {suggestions.map((suggestion, index) => (
                              <CommandItem
                                key={index}
                                onSelect={() => {
                                  setQuery(suggestion)
                                  setShowSuggestions(false)
                                  handleSubmit(new Event('submit') as any)
                                }}
                                className="cursor-pointer"
                              >
                                <Search className="h-4 w-4 mr-2 text-muted-foreground" />
                                {suggestion}
                              </CommandItem>
                            ))}
                          </CommandGroup>
                        </CommandList>
                      </Command>
                    </div>
                  )}
                </div>
              </form>
            </div>
          </div>
        ) : (
        <div className="p-6 space-y-4 relative">
          {isLoading && (
            <div className="absolute inset-0 bg-background/60 backdrop-blur-sm z-10 rounded-lg flex items-center justify-center">
              <div className="flex flex-col items-center gap-2">
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
                <p className="text-xs text-muted-foreground">
                  {agentMode ? "AI Agent is analyzing..." : "Searching..."}
                </p>
              </div>
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-4">
            {effectiveMode === "qa" && totalPapers > 0 && (
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
              {effectiveMode === "qa" || effectiveMode === "summary" ? (
                <Textarea
                  ref={textareaRef}
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value)
                    setShowSuggestions(true)
                  }}
                  onFocus={() => setShowSuggestions(true)}
                  onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                  placeholder={effectivePlaceholder}
                  className="min-h-[100px] resize-none pr-12"
                  disabled={isLoading}
                />
              ) : (
                <Input
                  ref={searchInputRef}
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value)
                    setShowSuggestions(true)
                  }}
                  onFocus={() => setShowSuggestions(true)}
                  onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
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
                  (effectiveMode === "summary" && selectedPapers.size === 0) ||
                  (effectiveMode === "qa" && qaMode === "single-paper" && selectedPapers.size === 0)
                }
                className="absolute right-2 top-2"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : effectiveMode === "paper-finder" ? (
                  <Search className="h-4 w-4" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>

              {/* Search suggestions dropdown */}
              {showSuggestions && suggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-popover border border-border rounded-md shadow-lg z-50">
                  <Command>
                    <CommandList className="max-h-48">
                      <CommandGroup>
                        {suggestions.map((suggestion, index) => (
                          <CommandItem
                            key={index}
                            onSelect={() => {
                              setQuery(suggestion)
                              setShowSuggestions(false)
                              if (effectiveMode === 'paper-finder') {
                                handleSubmit(new Event('submit') as any)
                              }
                            }}
                            className="cursor-pointer"
                          >
                            <Search className="h-4 w-4 mr-2 text-muted-foreground" />
                            {suggestion}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </div>
              )}
            </div>

            {effectiveMode === "paper-finder" && (
              <div className="flex items-center gap-2 flex-wrap">
                {/* Filter Presets */}
                <div className="flex items-center gap-2 flex-wrap">
                  {FILTER_PRESETS.map((preset, index) => {
                    const colors = [
                      "from-rose-100 to-pink-100 hover:from-rose-200 hover:to-pink-200 text-rose-700 border-rose-200",
                      "from-blue-100 to-indigo-100 hover:from-blue-200 hover:to-indigo-200 text-blue-700 border-blue-200",
                      "from-emerald-100 to-green-100 hover:from-emerald-200 hover:to-green-200 text-emerald-700 border-emerald-200",
                      "from-amber-100 to-yellow-100 hover:from-amber-200 hover:to-yellow-200 text-amber-700 border-amber-200",
                      "from-purple-100 to-violet-100 hover:from-purple-200 hover:to-violet-200 text-purple-700 border-purple-200"
                    ]
                    return (
                      <Button
                        key={preset.name}
                        variant="outline"
                        size="sm"
                        onClick={() => applyFilterPreset(preset)}
                        className={cn(
                          "text-xs h-8 font-semibold bg-gradient-to-r transition-all duration-300 shadow-sm hover:shadow-md",
                          colors[index % colors.length]
                        )}
                      >
                        {preset.name}
                      </Button>
                    )
                  })}
                </div>

                <Popover open={showFilters} onOpenChange={setShowFilters}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      className={cn(
                        "gap-2 h-8 font-semibold transition-all duration-300 shadow-sm hover:shadow-md",
                        hasActiveFilters
                          ? "bg-gradient-to-r from-purple-100 to-blue-100 border-purple-300 text-purple-700 hover:from-purple-200 hover:to-blue-200"
                          : "bg-gradient-to-r from-slate-50 to-white hover:from-slate-100 hover:to-slate-50 border-slate-200 text-slate-700"
                      )}
                    >
                      <Filter className="h-4 w-4" />
                      Custom Filters
                      {hasActiveFilters && (
                        <Badge className="ml-1 h-4 px-2 text-xs bg-gradient-to-r from-purple-500 to-blue-500 text-white border-none shadow-sm">
                          {
                            Object.values(filters).filter((v) => (Array.isArray(v) ? v.length > 0 : v !== undefined && v !== false))
                              .length
                          }
                        </Badge>
                      )}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-96 p-4" align="start">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium">Custom Filters</h4>
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

                      <div className="space-y-2">
                        <label className="text-sm font-medium flex items-center gap-2">
                          <CheckSquare className="h-4 w-4" />
                          Open Access
                        </label>
                        <div className="flex items-center space-x-2">
                          <Switch
                            checked={filters.openAccess || false}
                            onCheckedChange={(checked) =>
                              setFilters((prev) => ({
                                ...prev,
                                openAccess: checked || undefined,
                              }))
                            }
                          />
                          <span className="text-sm text-muted-foreground">Only show freely accessible papers</span>
                        </div>
                      </div>
                    </div>
                  </PopoverContent>
                </Popover>

                {hasActiveFilters && (
                  <div className="flex flex-wrap gap-2 mt-3">
                    {filters.dateRange?.from && (
                      <Badge className="text-xs bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-700 border border-blue-200 shadow-sm font-medium">
                        📅 From {filters.dateRange.from}
                        <button
                          onClick={() => setFilters(prev => ({
                            ...prev,
                            dateRange: prev.dateRange?.to ? { from: '', to: prev.dateRange.to } : undefined
                          }))}
                          className="ml-2 hover:bg-blue-200 rounded-full p-0.5 transition-colors"
                        >
                          ×
                        </button>
                      </Badge>
                    )}
                    {filters.dateRange?.to && (
                      <Badge className="text-xs bg-gradient-to-r from-indigo-100 to-purple-100 text-indigo-700 border border-indigo-200 shadow-sm font-medium">
                        📅 To {filters.dateRange.to}
                        <button
                          onClick={() => setFilters(prev => ({
                            ...prev,
                            dateRange: prev.dateRange?.from ? { from: prev.dateRange.from, to: '' } : undefined
                          }))}
                          className="ml-2 hover:bg-indigo-200 rounded-full p-0.5 transition-colors"
                        >
                          ×
                        </button>
                      </Badge>
                    )}
                    {filters.minCitations && (
                      <Badge className="text-xs bg-gradient-to-r from-amber-100 to-orange-100 text-amber-700 border border-amber-200 shadow-sm font-medium">
                        📊 {filters.minCitations}+ citations
                        <button
                          onClick={() => setFilters(prev => ({ ...prev, minCitations: undefined }))}
                          className="ml-2 hover:bg-amber-200 rounded-full p-0.5 transition-colors"
                        >
                          ×
                        </button>
                      </Badge>
                    )}
                    {filters.openAccess && (
                      <Badge className="text-xs bg-gradient-to-r from-emerald-100 to-green-100 text-emerald-700 border border-emerald-200 shadow-sm font-medium">
                        🔓 Open Access
                        <button
                          onClick={() => setFilters(prev => ({ ...prev, openAccess: undefined }))}
                          className="ml-2 hover:bg-emerald-200 rounded-full p-0.5 transition-colors"
                        >
                          ×
                        </button>
                      </Badge>
                    )}
                  </div>
                )}
              </div>
            )}
          </form>

          {!hideQuickActions && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-muted-foreground">Quick Actions</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {effectiveMode === "paper-finder" && (
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

                {effectiveMode === "qa" && (
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

                {effectiveMode === "summary" && (
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
          )}
        </div>
        )}
      </div>
    </div>
  )
}

export const MainSearchInterface = memo(MainSearchInterfaceComponent)

// Export types
export type { MainSearchInterfaceProps }
