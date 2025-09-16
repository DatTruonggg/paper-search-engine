"use client"

import type { SearchMode, SearchFilters } from "@/components/main-search-interface"
import type { PaperResult } from "@/components/paper-results-panel"
import { apiFetch } from "@/lib/apiClient"

export interface SearchResponse {
  contextId: string
  papers: PaperResult[]
  total: number
  page: number
  pageSize: number
}

export interface QAResponse {
  answer: string
  sources: PaperResult[]
  evidence: Array<{ paperId: string; locator?: { page?: number; paragraph?: number }; snippet?: string; score?: number }>
}

export interface SummaryResponse {
  summary: string
  keyPapers: PaperResult[]
  topics: Array<{ name: string; papers: string[] }>
}

export interface RetrievalParams {
  topK?: number
  rerankK?: number
  maxChunksPerDoc?: number
}

export class ResearchService {
  async searchPapers(query: string, filters?: SearchFilters, page = 1, pageSize = 20, sort: string = "relevance") {
    return apiFetch<SearchResponse>(`/v1/search`, {
      method: 'POST',
      body: JSON.stringify({ query, filters, page, pageSize, sort }),
    })
  }

  async bookmark(paperId: string, on: boolean) {
    const method = on ? 'PUT' : 'DELETE'
    return apiFetch(`/v1/bookmarks/${encodeURIComponent(paperId)}`, { method })
  }

  async sessionsList() {
    return apiFetch<{ sessions: any[] }>(`/v1/sessions`, { method: 'GET' })
  }

  async sessionsCreate(payload: { title: string; mode: SearchMode }) {
    return apiFetch<{ id: string }>(`/v1/sessions`, { method: 'POST', body: JSON.stringify(payload) })
  }

  async sessionsPatch(id: string, patch: Partial<{ title: string; mode: SearchMode }>) {
    return apiFetch(`/v1/sessions/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(patch) })
  }

  // --- Suggest ---
  async suggest(q: string) {
    return apiFetch<{ suggestions: string[] }>(`/v1/suggest?q=${encodeURIComponent(q)}`, { method: 'GET' })
  }

  // --- QA ---
  async qaSelected(params: {
    question: string
    paperIds: string[]
    retrieval?: RetrievalParams
    citationStyle?: 'APA' | 'IEEE' | 'MLA'
    conversationId?: string
  }) {
    return apiFetch<QAResponse>(`/v1/qa`, {
      method: 'POST',
      body: JSON.stringify({
        mode: 'selected',
        question: params.question,
        paperIds: params.paperIds,
        retrieval: params.retrieval,
        citationStyle: params.citationStyle,
        conversationId: params.conversationId,
      }),
    })
  }

  async qaAll(params: {
    question: string
    searchContextId: string
    retrieval?: RetrievalParams
    citationStyle?: 'APA' | 'IEEE' | 'MLA'
    conversationId?: string
  }) {
    return apiFetch<QAResponse>(`/v1/qa`, {
      method: 'POST',
      body: JSON.stringify({
        mode: 'all',
        searchContextId: params.searchContextId,
        question: params.question,
        retrieval: params.retrieval,
        citationStyle: params.citationStyle,
        conversationId: params.conversationId,
      }),
    })
  }

  // --- Summary ---
  async summarySelected(params: {
    paperIds: string[]
    topic?: string
    scope?: string[]
    length?: 'short' | 'medium' | 'long'
    citationStyle?: 'APA' | 'IEEE' | 'MLA'
  }) {
    return apiFetch<SummaryResponse>(`/v1/summary`, {
      method: 'POST',
      body: JSON.stringify({
        mode: 'selected',
        paperIds: params.paperIds,
        topic: params.topic,
        scope: params.scope,
        length: params.length,
        citationStyle: params.citationStyle,
      }),
    })
  }

  async summaryAll(params: {
    searchContextId: string
    topic?: string
    scope?: string[]
    length?: 'short' | 'medium' | 'long'
    citationStyle?: 'APA' | 'IEEE' | 'MLA'
  }) {
    return apiFetch<SummaryResponse>(`/v1/summary`, {
      method: 'POST',
      body: JSON.stringify({
        mode: 'all',
        searchContextId: params.searchContextId,
        topic: params.topic,
        scope: params.scope,
        length: params.length,
        citationStyle: params.citationStyle,
      }),
    })
  }

  // --- Documents ---
  async documentsList(params?: { q?: string; tags?: string[]; status?: string; page?: number; pageSize?: number }) {
    const search = new URLSearchParams()
    if (params?.q) search.set('q', params.q)
    if (params?.status) search.set('status', params.status)
    if (params?.page) search.set('page', String(params.page))
    if (params?.pageSize) search.set('pageSize', String(params.pageSize))
    if (params?.tags && params.tags.length) search.set('tags', params.tags.join(','))
    return apiFetch<{ items: any[]; page: number; pageSize: number; total: number }>(`/v1/documents?${search.toString()}`, { method: 'GET' })
  }

  async documentsGet(id: string) {
    return apiFetch(`/v1/documents/${encodeURIComponent(id)}`, { method: 'GET' })
  }

  async documentsDelete(id: string) {
    return apiFetch(`/v1/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
  }

  async documentsIngestUrl(url: string, opts?: { name?: string; tags?: string[] }) {
    return apiFetch(`/v1/documents:ingest-url`, {
      method: 'POST',
      body: JSON.stringify({ url, name: opts?.name, tags: opts?.tags }),
    })
  }
}
