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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"
import { Switch } from "@/components/ui/switch"
import { Command, CommandGroup, CommandItem, CommandList } from "@/components/ui/command"
import { useSearchShortcuts } from "@/hooks/use-keyboard-shortcuts"
import { useSearchSuggestions } from "@/hooks/use-search-suggestions"
import { useDebounce } from "@/hooks/use-debounce"
// Removed advanced filter modal; keep simple filters only

export type SearchMode = "paper-finder" | "qa" | "summary"
export type QAMode = "single-paper" | "all-papers"

interface MainSearchInterfaceProps {
  mode: SearchMode
  onModeChange: (mode: SearchMode) => void
  onSearch: (
    query: string,
    filters?: SearchFilters,
    qaMode?: QAMode,
    isAgentMode?: boolean,
    backendMode?: 'fulltext' | 'hybrid' | 'semantic',
  ) => void
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
  /**
   * When true, disables inline suggestions and related API calls
   */
  disableSuggestions?: boolean
  /**
   * When provided, forces the agent mode to be active (overrides internal state)
   */
  forceAgentMode?: boolean
}

export interface SearchFilters {
  dateRange?: {
    from: string
    to: string
  }
  authors?: string[]
  // Other fields intentionally unsupported in UI per spec
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
  disableSuggestions = true,
  forceAgentMode = false,
}: MainSearchInterfaceProps) {
  const [query, setQuery] = useState("")
  const [filters, setFilters] = useState<SearchFilters>({})
  const [qaMode, setQaMode] = useState<QAMode>("single-paper")
  const [agentMode, setAgentMode] = useState(false)
  // Use forced agent mode if provided, otherwise use internal state
  const effectiveAgentMode = forceAgentMode || agentMode
  const [agentSteps, setAgentSteps] = useState<string[]>([])
  const [backendSearchMode, setBackendSearchMode] = useState<'fulltext' | 'hybrid' | 'semantic'>("fulltext")
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

  // Restore loading chip if a search is in progress (e.g., after navigation)
  useEffect(() => {
    try {
      const raw = localStorage.getItem('pse:searchProgress')
      if (!raw) return
      const st = JSON.parse(raw) as { inProgress?: boolean; isAgent?: boolean; stepIndex?: number }
      if (st?.inProgress && st?.isAgent) {
        setAgentMode(true)
        const steps = [
          'Analyzing query',
          'Planning retrieval strategy',
          'Fetching candidate papers',
          'Extracting evidence',
          'Ranking results',
        ]
        const idx = Number.isFinite(st.stepIndex) ? (st.stepIndex as number) : 0
        setAgentSteps(steps.slice(0, Math.max(1, idx + 1)))
        // continue timers to end with longer timing
        const schedule = [15000, 30000, 50000, 75000]
        schedule.forEach((ms, i) => {
          window.setTimeout(() => setAgentSteps((s) => (s.includes(steps[i + 1]) ? s : s.concat(steps[i + 1]))), ms)
        })
      }
    } catch {}
  }, [])

  // Debounce query for suggestions (optional)
  const debouncedQuery = useDebounce(query, 300)
  const suggestions = disableSuggestions ? [] : useSearchSuggestions(debouncedQuery, showSuggestions).suggestions

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
      if (effectiveAgentMode) {
        // Force fake steps to always show when agent mode is enabled
        const steps = [
          'Analyzing query',
          'Planning retrieval strategy',
          'Fetching candidate papers',
          'Extracting evidence',
          'Ranking results',
        ]
        // Always start with the first step immediately
        setAgentSteps([steps[0]])

        // Store search progress state
        try {
          localStorage.setItem('pse:searchProgress', JSON.stringify({
            inProgress: true,
            isAgent: true,
            stepIndex: 0
          }))
        } catch {}

        // Force the remaining fake steps to show with longer timing
        const schedule = [15000, 30000, 50000, 75000]
        schedule.forEach((ms, i) => {
          window.setTimeout(() => {
            setAgentSteps((prevSteps) => {
              // Always add the next step, ensuring we show all fake steps
              const nextStep = steps[i + 1]
              if (nextStep && !prevSteps.includes(nextStep)) {
                return [...prevSteps, nextStep]
              }
              return prevSteps
            })
            try {
              const currentState = JSON.parse(localStorage.getItem('pse:searchProgress') || '{}')
              localStorage.setItem('pse:searchProgress', JSON.stringify({
                ...currentState,
                stepIndex: i + 1
              }))
            } catch {}
          }, ms)
        })
      } else {
        // Clear agent steps if not in agent mode
        setAgentSteps([])
      }
      onSearch(query.trim(), filters, effectiveMode === "qa" ? qaMode : undefined, effectiveAgentMode, backendSearchMode)
    }
  }, [query, filters, effectiveMode, qaMode, effectiveAgentMode, backendSearchMode, onSearch])

  const getModeConfig = (searchMode: SearchMode) => {
    switch (searchMode) {
      case "paper-finder":
        return {
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

  const hasActiveFilters = Boolean(
    (filters.dateRange?.from && filters.dateRange.from.trim() !== '') ||
    (filters.dateRange?.to && filters.dateRange.to.trim() !== '') ||
    (filters.authors && filters.authors.length > 0)
  )

  const applyFilterPreset = useCallback((preset: typeof FILTER_PRESETS[0]) => {
    setFilters(preset.filters)
  }, [])

  const isHero = effectiveMode === "paper-finder" && centerOnEmpty && totalPapers === 0

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Mode Selection */}
      <div className="border-b border-border bg-card">
        <div className="p-4">
          {!hideModeSelector && (
            <div className="flex items-center gap-2 mb-4">
              <div className="flex rounded-lg border border-slate-200 p-1 bg-white shadow-sm">
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
                        "flex items-center gap-2 px-3 py-2 transition-colors",
                        isActive
                          ? "bg-green-600 text-white hover:bg-green-700"
                          : "text-slate-600 hover:bg-green-50 hover:text-green-700",
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
                <h2 className="text-2xl font-bold bg-gradient-to-r from-green-700 to-emerald-600 bg-clip-text text-transparent mb-1">{currentConfig.label}</h2>
                <p className="text-sm text-slate-600">{currentConfig.description}</p>
              </div>
              
            </div>

            {effectiveAgentMode && (
              <div className="mb-3 p-3 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg shadow-sm">
                <div className="flex items-center gap-2 text-sm">
                  <Zap className="h-4 w-4 text-green-600" />
                  <span className="font-semibold text-green-700">
                    AI Agent Mode Active
                  </span>
                </div>
                <p className="text-xs text-slate-600 mt-1">
                  The AI agent will analyze your query and perform intelligent multi-step searches
                </p>
              </div>
            )}
          </div>

          {(effectiveMode === "summary" || effectiveMode === "qa") && totalPapers > 0 && (
            <div className="mb-4 p-3 bg-gradient-to-r from-green-50/50 to-emerald-50/50 rounded-lg border border-green-100">
              <div className="flex items-center gap-2 text-sm">
                <CheckSquare className="h-4 w-4 text-green-600" />
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
          <div className="flex-1 flex items-center justify-center relative bg-gradient-to-br from-green-50/30 to-emerald-50/20"
          >
            {isLoading && (
              <div className="absolute inset-0 bg-white/80 backdrop-blur-sm z-10 flex items-center justify-center">
                <div className="flex flex-col items-center gap-3 p-6 rounded-lg bg-white shadow-lg border border-green-100">
                  <Loader2 className="h-6 w-6 animate-spin text-green-600" />
                  {effectiveAgentMode ? (
                    <div className="w-full max-w-sm">
                      <p className="text-lg font-semibold text-slate-700 mb-2">Agent progress</p>
                      <ul className="text-xs text-slate-700 space-y-1">
                        {agentSteps.map((step, idx) => (
                          <li key={idx} className="flex items-center gap-2">
                            <span className="inline-block w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                            {step}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : (
                    <p className="text-sm font-medium text-slate-700">Searching papers...</p>
                  )}
                </div>
              </div>
            )}
            <div className="max-w-2xl w-full px-6 py-8">
              <div className="text-center mb-10">
                <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-green-700 to-emerald-700 bg-clip-text text-transparent mb-4">
                  Research Discovery Hub
                </h1>
                <p className="text-slate-600 mt-3 text-lg font-light">
                  Discover and explore academic papers with intelligent search
                </p>
              </div>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="relative">
                  <Input
                    ref={searchInputRef}
                    value={query}
                    onChange={(e) => {
                      setQuery(e.target.value)
                      if (!disableSuggestions) setShowSuggestions(true)
                    }}
                    onFocus={() => !disableSuggestions && setShowSuggestions(true)}
                    onBlur={() => !disableSuggestions && setTimeout(() => setShowSuggestions(false), 200)}
                    placeholder={effectivePlaceholder}
                    className="pr-12 h-12 lg:h-14 text-base lg:text-lg border-green-200 focus:border-green-400 focus:ring-2 focus:ring-green-400/20 shadow-sm hover:shadow-md transition-shadow"
                    disabled={isLoading}
                  />
                  <Button
                    type="submit"
                    size="sm"
                    disabled={!query.trim() || isLoading}
                    className="absolute right-2 top-2 lg:top-2.5 h-8 lg:h-9 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 border-0 text-white shadow-md transition-all"
                  >
                    {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                  </Button>

                  {/* Search suggestions dropdown */}
                  {!disableSuggestions && showSuggestions && suggestions.length > 0 && (
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
          {/* Keep Research Discovery Hub banner even when results are shown */}
          <div className="mb-4">
            <h1 className="text-center text-4xl lg:text-5xl font-bold bg-gradient-to-r from-green-700 to-emerald-600 bg-clip-text text-transparent">Research Discovery Hub</h1>
            <p className="text-center text-slate-600 mt-2 text-lg font-light">Discover and explore academic papers with intelligent search</p>
          </div>
          {isLoading && (
            <div className="absolute inset-0 bg-background/60 backdrop-blur-sm z-10 rounded-lg flex items-center justify-center">
              <div className="flex flex-col items-center gap-2">
                <Loader2 className="h-5 w-5 animate-spin text-green-600" />
                {effectiveAgentMode ? (
                  <div className="w-full max-w-sm">
                    <p className="text-sm font-semibold text-slate-700 mb-1">Agent progress</p>
                    <ul className="text-xs text-slate-600 space-y-1">
                      {agentSteps.map((step, idx) => (
                        <li key={idx} className="flex items-center gap-2">
                          <span className="inline-block w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                          {step}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">Searching...</p>
                )}
              </div>
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-4">
            {isLoading && effectiveAgentMode && (
              <div className="flex items-center gap-2 text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-md px-2 py-1 w-fit">
                <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                Agent progress: {agentSteps[agentSteps.length - 1] || 'Starting...'}
              </div>
            )}
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
                    if (!disableSuggestions) setShowSuggestions(true)
                  }}
                  onFocus={() => !disableSuggestions && setShowSuggestions(true)}
                  onBlur={() => !disableSuggestions && setTimeout(() => setShowSuggestions(false), 200)}
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
                    if (!disableSuggestions) setShowSuggestions(true)
                  }}
                  onFocus={() => !disableSuggestions && setShowSuggestions(true)}
                  onBlur={() => !disableSuggestions && setTimeout(() => setShowSuggestions(false), 200)}
                  placeholder={effectivePlaceholder}
                  className="pr-12 h-12 lg:h-14 text-base lg:text-lg border-green-200 focus:border-green-400 focus:ring-2 focus:ring-green-400/20 shadow-sm hover:shadow-md transition-shadow"
                  disabled={isLoading}
                />
              )}

              <Button
                type="submit"
                size="sm"
                disabled={
                  !query.trim() ||
                  isLoading ||
                  (effectiveMode === "summary" && selectedPapers.size === 0)
                }
                className="absolute right-2 top-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white border-0 shadow-sm transition-all"
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
              {!disableSuggestions && showSuggestions && suggestions.length > 0 && (
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
              <div className="space-y-3">
                <div className="p-2 border rounded-md">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2 text-sm font-medium">
                      <Filter className="h-4 w-4" /> Filters
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] text-slate-600">Mode</span>
                      <Select value={backendSearchMode} onValueChange={(v: 'fulltext'|'hybrid'|'semantic') => setBackendSearchMode(v)}>
                        <SelectTrigger className="h-8 w-28 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="fulltext">Fulltext</SelectItem>
                          <SelectItem value="hybrid">Hybrid</SelectItem>
                          <SelectItem value="semantic">Semantic</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    <div>
                      <label className="text-xs text-slate-600">Year From</label>
                      <Input
                        placeholder="From"
                        type="number"
                        min="1900"
                        max="2100"
                        value={filters.dateRange?.from || ""}
                        className="h-8"
                        onChange={(e) =>
                          setFilters((prev) => ({
                            ...prev,
                            dateRange: { ...prev.dateRange, from: e.target.value, to: prev.dateRange?.to || "" },
                          }))
                        }
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-600">Year To</label>
                      <Input
                        placeholder="To"
                        type="number"
                        min="1900"
                        max="2100"
                        value={filters.dateRange?.to || ""}
                        className="h-8"
                        onChange={(e) =>
                          setFilters((prev) => ({
                            ...prev,
                            dateRange: { ...prev.dateRange, to: e.target.value, from: prev.dateRange?.from || "" },
                          }))
                        }
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-600">Author</label>
                      <Input
                        placeholder="Author name"
                        value={(filters.authors && filters.authors[0]) || ""}
                        className="h-8"
                        onChange={(e) => setFilters((prev) => ({ ...prev, authors: e.target.value ? [e.target.value] : [] }))}
                      />
                    </div>
                  </div>
                </div>

                {hasActiveFilters && (
                  <div className="flex flex-wrap gap-2">
                    {filters.dateRange?.from && (
                      <Badge className="text-xs bg-green-50 text-green-700 border border-green-200 font-medium">
                        From {filters.dateRange.from}
                        <button
                          onClick={() => setFilters(prev => ({
                            ...prev,
                            dateRange: prev.dateRange?.to ? { from: '', to: prev.dateRange.to } : undefined
                          }))}
                          className="ml-2 hover:bg-green-200 rounded-full p-0.5 transition-colors"
                        >
                          ×
                        </button>
                      </Badge>
                    )}
                    {filters.dateRange?.to && (
                      <Badge className="text-xs bg-green-50 text-green-700 border border-green-200 font-medium">
                        To {filters.dateRange.to}
                        <button
                          onClick={() => setFilters(prev => ({
                            ...prev,
                            dateRange: prev.dateRange?.from ? { from: prev.dateRange.from, to: '' } : undefined
                          }))}
                          className="ml-2 hover:bg-green-200 rounded-full p-0.5 transition-colors"
                        >
                          ×
                        </button>
                      </Badge>
                    )}
                    {filters.authors && filters.authors[0] && (
                      <Badge className="text-xs bg-green-50 text-green-700 border border-green-200 font-medium">
                        {filters.authors[0]}
                        <button
                          onClick={() => setFilters(prev => ({ ...prev, authors: [] }))}
                          className="ml-2 hover:bg-green-200 rounded-full p-0.5 transition-colors"
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
                      onClick={() => setQuery("")}
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
