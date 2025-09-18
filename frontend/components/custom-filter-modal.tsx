"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Calendar,
  BookOpen,
  CheckSquare,
  Users,
  Tags,
  Filter,
  X,
  Plus,
} from "lucide-react"
import { cn } from "@/lib/utils"

export interface SearchFilters {
  dateRange?: {
    from: string
    to: string
  }
  authors?: string[]
  venues?: string[]
  minCitations?: number
  keywords?: string[]
  domains?: string[]
  openAccess?: boolean
}

interface CustomFilterModalProps {
  filters: SearchFilters
  onFiltersChange: (filters: SearchFilters) => void
  hasActiveFilters: boolean
  onClearFilters: () => void
}

// Common domain/category suggestions
const COMMON_DOMAINS = [
  "Computer Science",
  "Machine Learning",
  "Artificial Intelligence",
  "Natural Language Processing",
  "Computer Vision",
  "Data Science",
  "Software Engineering",
  "Distributed Systems",
  "Networks",
  "Security",
  "Algorithms",
  "Theory",
  "Human-Computer Interaction",
  "Databases",
  "Programming Languages",
]

export function CustomFilterModal({
  filters,
  onFiltersChange,
  hasActiveFilters,
  onClearFilters,
}: CustomFilterModalProps) {
  const [open, setOpen] = useState(false)
  const [authorInput, setAuthorInput] = useState("")
  const [domainInput, setDomainInput] = useState("")

  const addAuthor = () => {
    if (authorInput.trim() && !filters.authors?.includes(authorInput.trim())) {
      onFiltersChange({
        ...filters,
        authors: [...(filters.authors || []), authorInput.trim()],
      })
      setAuthorInput("")
    }
  }

  const removeAuthor = (author: string) => {
    onFiltersChange({
      ...filters,
      authors: filters.authors?.filter((a) => a !== author),
    })
  }

  const addDomain = (domain: string) => {
    if (domain.trim() && !filters.domains?.includes(domain.trim())) {
      onFiltersChange({
        ...filters,
        domains: [...(filters.domains || []), domain.trim()],
      })
      setDomainInput("")
    }
  }

  const removeDomain = (domain: string) => {
    onFiltersChange({
      ...filters,
      domains: filters.domains?.filter((d) => d !== domain),
    })
  }

  const handleDateRangeChange = (field: 'from' | 'to', value: string) => {
    onFiltersChange({
      ...filters,
      dateRange: {
        ...filters.dateRange,
        [field]: value,
        ...(field === 'from' ? { to: filters.dateRange?.to || "" } : { from: filters.dateRange?.from || "" }),
      },
    })
  }

  const activeFilterCount = Object.values(filters).filter((v) =>
    Array.isArray(v) ? v.length > 0 : v !== undefined && v !== false
  ).length

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={cn(
            "gap-2 h-8 font-medium transition-all duration-200",
            hasActiveFilters
              ? "bg-green-100 border-green-400 text-green-700 hover:bg-green-200"
              : "bg-white hover:bg-green-50 border-slate-200 text-slate-700"
          )}
        >
          <Filter className="h-4 w-4" />
          Custom Filters
          {hasActiveFilters && (
            <Badge className="ml-1 h-4 px-2 text-xs bg-green-600 text-white border-none">
              {activeFilterCount}
            </Badge>
          )}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px] max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Custom Filters
          </DialogTitle>
          <DialogDescription>
            Refine your search with specific criteria for year, domain, and authors.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Publication Year */}
          <div className="space-y-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              <Calendar className="h-4 w-4 text-green-600" />
              Publication Year
            </Label>
            <div className="flex gap-2">
              <div className="flex-1">
                <Input
                  placeholder="From year"
                  type="number"
                  min="1900"
                  max="2024"
                  value={filters.dateRange?.from || ""}
                  onChange={(e) => handleDateRangeChange('from', e.target.value)}
                  className="text-sm"
                />
              </div>
              <div className="flex-1">
                <Input
                  placeholder="To year"
                  type="number"
                  min="1900"
                  max="2024"
                  value={filters.dateRange?.to || ""}
                  onChange={(e) => handleDateRangeChange('to', e.target.value)}
                  className="text-sm"
                />
              </div>
            </div>
          </div>

          {/* Domains/Categories */}
          <div className="space-y-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              <Tags className="h-4 w-4 text-blue-600" />
              Domains/Categories
            </Label>

            {/* Quick domain suggestions */}
            <div className="space-y-2">
              <div className="text-xs text-slate-600 mb-2">Quick add:</div>
              <div className="flex flex-wrap gap-1">
                {COMMON_DOMAINS.slice(0, 8).map((domain) => (
                  <Button
                    key={domain}
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-xs hover:bg-blue-50 hover:text-blue-700 border border-slate-200 hover:border-blue-300"
                    onClick={() => addDomain(domain)}
                    disabled={filters.domains?.includes(domain)}
                  >
                    {domain}
                  </Button>
                ))}
              </div>
            </div>

            {/* Custom domain input */}
            <div className="flex gap-2">
              <Input
                placeholder="Add custom domain..."
                value={domainInput}
                onChange={(e) => setDomainInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault()
                    addDomain(domainInput)
                  }
                }}
                className="text-sm"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => addDomain(domainInput)}
                disabled={!domainInput.trim()}
                className="px-3"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>

            {/* Selected domains */}
            {filters.domains && filters.domains.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {filters.domains.map((domain) => (
                  <Badge
                    key={domain}
                    variant="secondary"
                    className="text-xs bg-blue-100 text-blue-700 hover:bg-blue-200 pr-1"
                  >
                    {domain}
                    <button
                      onClick={() => removeDomain(domain)}
                      className="ml-1 hover:bg-blue-300 rounded-full p-0.5"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Authors */}
          <div className="space-y-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              <Users className="h-4 w-4 text-purple-600" />
              Authors
            </Label>
            <div className="flex gap-2">
              <Input
                placeholder="Add author name..."
                value={authorInput}
                onChange={(e) => setAuthorInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault()
                    addAuthor()
                  }
                }}
                className="text-sm"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={addAuthor}
                disabled={!authorInput.trim()}
                className="px-3"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>

            {/* Selected authors */}
            {filters.authors && filters.authors.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {filters.authors.map((author) => (
                  <Badge
                    key={author}
                    variant="secondary"
                    className="text-xs bg-purple-100 text-purple-700 hover:bg-purple-200 pr-1"
                  >
                    {author}
                    <button
                      onClick={() => removeAuthor(author)}
                      className="ml-1 hover:bg-purple-300 rounded-full p-0.5"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Minimum Citations */}
          <div className="space-y-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              <BookOpen className="h-4 w-4 text-orange-600" />
              Minimum Citations
            </Label>
            <Select
              value={filters.minCitations?.toString() || "any"}
              onValueChange={(value) =>
                onFiltersChange({
                  ...filters,
                  minCitations: value === "any" ? undefined : Number.parseInt(value),
                })
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

          {/* Open Access */}
          <div className="space-y-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              <CheckSquare className="h-4 w-4 text-green-600" />
              Open Access
            </Label>
            <div className="flex items-center space-x-2">
              <Switch
                checked={filters.openAccess || false}
                onCheckedChange={(checked) =>
                  onFiltersChange({
                    ...filters,
                    openAccess: checked || undefined,
                  })
                }
              />
              <span className="text-sm text-slate-600">Only show freely accessible papers</span>
            </div>
          </div>
        </div>

        <DialogFooter className="flex items-center justify-between">
          <div className="flex gap-2">
            {hasActiveFilters && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  onClearFilters()
                  setAuthorInput("")
                  setDomainInput("")
                }}
                className="text-slate-600 hover:text-slate-800"
              >
                Clear all
              </Button>
            )}
          </div>
          <Button onClick={() => setOpen(false)} className="bg-green-600 hover:bg-green-700">
            Apply Filters
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}