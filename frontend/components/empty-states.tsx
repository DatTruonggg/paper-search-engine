import { BookOpen, Search, Lightbulb, Zap, TrendingUp } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface EmptyStateProps {
  title: string
  description: string
  icon: React.ReactNode
  actions?: Array<{
    label: string
    action: () => void
    variant?: 'default' | 'outline' | 'ghost'
  }>
  suggestions?: string[]
  onSuggestionClick?: (suggestion: string) => void
}

export function EmptyState({
  title,
  description,
  icon,
  actions,
  suggestions,
  onSuggestionClick
}: EmptyStateProps) {
  return (
    <div className="flex items-center justify-center min-h-[400px] p-8">
      <Card className="w-full max-w-md text-center bg-gradient-to-br from-white via-slate-50 to-slate-100 border-2 border-slate-200 shadow-2xl">
        <CardHeader>
          <div className="mx-auto w-20 h-20 bg-gradient-to-br from-purple-100 via-blue-100 to-indigo-100 rounded-full flex items-center justify-center mb-6 shadow-lg border border-white">
            {icon}
          </div>
          <CardTitle className="text-2xl font-bold bg-gradient-to-r from-slate-700 to-slate-600 bg-clip-text text-transparent">{title}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-slate-600 font-medium leading-relaxed">{description}</p>

          {suggestions && suggestions.length > 0 && (
            <div className="space-y-3">
              <p className="text-sm font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">✨ Try searching for:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {suggestions.map((suggestion, index) => {
                  const colors = [
                    "bg-gradient-to-r from-rose-100 to-pink-100 hover:from-rose-200 hover:to-pink-200 text-rose-700 border-rose-200",
                    "bg-gradient-to-r from-blue-100 to-indigo-100 hover:from-blue-200 hover:to-indigo-200 text-blue-700 border-blue-200",
                    "bg-gradient-to-r from-emerald-100 to-green-100 hover:from-emerald-200 hover:to-green-200 text-emerald-700 border-emerald-200",
                    "bg-gradient-to-r from-amber-100 to-yellow-100 hover:from-amber-200 hover:to-yellow-200 text-amber-700 border-amber-200",
                    "bg-gradient-to-r from-purple-100 to-violet-100 hover:from-purple-200 hover:to-violet-200 text-purple-700 border-purple-200",
                    "bg-gradient-to-r from-cyan-100 to-teal-100 hover:from-cyan-200 hover:to-teal-200 text-cyan-700 border-cyan-200"
                  ]
                  return (
                    <Button
                      key={index}
                      variant="outline"
                      size="sm"
                      onClick={() => onSuggestionClick?.(suggestion)}
                      className={cn(
                        "text-xs h-8 font-semibold transition-all duration-300 shadow-sm hover:shadow-md",
                        colors[index % colors.length]
                      )}
                    >
                      {suggestion}
                    </Button>
                  )
                })}
              </div>
            </div>
          )}

          {actions && actions.length > 0 && (
            <div className="flex gap-3 justify-center pt-4">
              {actions.map((action, index) => (
                <Button
                  key={index}
                  variant={action.variant || 'default'}
                  onClick={action.action}
                  className={cn(
                    "font-semibold shadow-lg hover:shadow-xl transition-all duration-300",
                    action.variant === 'outline'
                      ? "bg-gradient-to-r from-slate-50 to-white hover:from-slate-100 hover:to-slate-50 border-slate-300 text-slate-700"
                      : "bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white"
                  )}
                >
                  {action.label}
                </Button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export function NoResultsFound({
  searchQuery,
  onSuggestionClick,
  onClearFilters
}: {
  searchQuery?: string
  onSuggestionClick?: (suggestion: string) => void
  onClearFilters?: () => void
}) {
  const suggestions = [
    'machine learning',
    'artificial intelligence',
    'deep learning',
    'neural networks',
    'computer vision'
  ]

  return (
    <EmptyState
      icon={<Search className="h-8 w-8 text-slate-500" />}
      title="No papers found"
      description={
        searchQuery
          ? `We couldn't find any papers matching "${searchQuery}". Try adjusting your search terms or filters.`
          : "Start searching to discover academic papers on your topic of interest."
      }
      suggestions={suggestions}
      onSuggestionClick={onSuggestionClick}
      actions={
        onClearFilters
          ? [
              {
                label: 'Clear filters',
                action: onClearFilters,
                variant: 'outline' as const
              }
            ]
          : undefined
      }
    />
  )
}

export function SearchWelcome({
  onSuggestionClick,
  showDiscoverText = false
}: {
  onSuggestionClick?: (suggestion: string) => void
  showDiscoverText?: boolean
}) {
  const popularTopics = [
    'transformer models',
    'computer vision',
    'reinforcement learning',
    'natural language processing',
    'quantum computing',
    'bioinformatics'
  ]

  return (
    <EmptyState
      icon={<BookOpen className="h-10 w-10 text-purple-600" />}
      title={showDiscoverText ? "Discover Academic Papers" : "Academic Papers"}
      description={showDiscoverText
        ? "Search through millions of academic papers to find the research that matters to you."
        : "Search for academic papers and research on your topics of interest."
      }
      suggestions={popularTopics}
      onSuggestionClick={onSuggestionClick}
    />
  )
}

export function LoadingError({ onRetry }: { onRetry?: () => void }) {
  return (
    <EmptyState
      icon={<Zap className="h-8 w-8 text-red-500" />}
      title="Unable to load papers"
      description="There was an issue loading the search results. This might be due to a network problem."
      actions={[
        {
          label: 'Try again',
          action: onRetry || (() => window.location.reload()),
        }
      ]}
    />
  )
}