import { useState, useEffect, useMemo } from "react"
import { ResearchService } from "@/lib/research-service"

const POPULAR_TOPICS = [
  "machine learning",
  "artificial intelligence",
  "deep learning",
  "neural networks",
  "computer vision",
  "natural language processing",
  "reinforcement learning",
  "transformers",
  "large language models",
  "computer science",
  "data science",
  "algorithms",
  "robotics",
  "blockchain",
  "cybersecurity",
  "quantum computing",
  "bioinformatics",
  "human-computer interaction",
  "software engineering",
  "distributed systems"
]

export function useSearchSuggestions(query: string, enabled: boolean = true) {
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const api = useMemo(() => new ResearchService(), [])

  // Get filtered popular topics based on query
  const popularSuggestions = useMemo(() => {
    if (!query || query.length < 2) return []

    return POPULAR_TOPICS
      .filter(topic => topic.toLowerCase().includes(query.toLowerCase()))
      .slice(0, 5)
  }, [query])

  useEffect(() => {
    if (!enabled || !query || query.length < 3) {
      setSuggestions(popularSuggestions)
      return
    }

    const timeoutId = setTimeout(async () => {
      setLoading(true)
      try {
        const response = await api.suggest(query)
        // Combine API suggestions with popular topics
        const combined = [
          ...response.suggestions.slice(0, 3),
          ...popularSuggestions.filter(topic =>
            !response.suggestions.some(suggestion =>
              suggestion.toLowerCase().includes(topic.toLowerCase())
            )
          )
        ].slice(0, 8)

        setSuggestions(combined)
      } catch (error) {
        // Fallback to popular suggestions on error
        setSuggestions(popularSuggestions)
      } finally {
        setLoading(false)
      }
    }, 300) // Debounce

    return () => clearTimeout(timeoutId)
  }, [query, enabled, api, popularSuggestions])

  return { suggestions, loading }
}