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
      <Card className="w-full max-w-md text-center bg-white border border-slate-200 shadow-lg">
        <CardHeader>
          <div className="mx-auto w-20 h-20 bg-gradient-to-br from-green-100 to-emerald-100 rounded-full flex items-center justify-center mb-6 shadow-md border border-white">
            {icon}
          </div>
          <CardTitle className="text-2xl font-bold text-slate-800">{title}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-slate-600 leading-relaxed">{description}</p>

          {suggestions && suggestions.length > 0 && (
            <div className="space-y-3">
              <p className="text-sm font-semibold text-green-700">Try searching for:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {suggestions.map((suggestion, index) => {
                  const colors = [
                    "bg-green-50 hover:bg-green-100 text-green-700 border-green-300",
                    "bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border-emerald-300",
                    "bg-teal-50 hover:bg-teal-100 text-teal-700 border-teal-300",
                    "bg-green-50 hover:bg-green-100 text-green-700 border-green-300",
                    "bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border-emerald-300",
                    "bg-teal-50 hover:bg-teal-100 text-teal-700 border-teal-300"
                  ]
                  return (
                    <Button
                      key={index}
                      variant="outline"
                      size="sm"
                      onClick={() => onSuggestionClick?.(suggestion)}
                      className={cn(
                        "text-xs h-8 font-medium transition-all duration-200 shadow-sm hover:shadow-md",
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
                    "font-medium shadow-sm hover:shadow-md transition-all duration-200",
                    action.variant === 'outline'
                      ? "bg-white hover:bg-green-50 border-slate-300 text-slate-700 hover:text-green-700 hover:border-green-300"
                      : "bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white border-0"
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
      icon={<BookOpen className="h-10 w-10 text-green-600" />}
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