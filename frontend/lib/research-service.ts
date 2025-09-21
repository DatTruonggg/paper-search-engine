"use client"

import type { SearchMode, SearchFilters } from "@/components/main-search-interface"
import type { PaperResult } from "@/components/paper-results-panel"
import { apiFetch } from "@/lib/apiClient"

// Backend search response models
export interface BackendSearchResponse {
  results: Array<{
    paper_id: string
    title: string
    authors: string[]
    abstract: string
    score: number
    categories: string[]
    publish_date: string
    word_count: number
    has_images: boolean
    pdf_size: number
  }>
  total: number
  query: string
  search_mode: string
}

export interface BackendAgentResponse {
  success: boolean
  query: string
  papers: Array<{
    paper_id: string
    title: string
    authors: string[]
    abstract: string
    score: number
    categories: string[]
    publish_date: string
    word_count: number
    has_images: boolean
    pdf_size: number
    evidence_sentences?: string[]
  }>
  total_found: number
  search_iterations: number
  error?: string
}

// Batch papers metadata (lightweight) response
export interface BackendPapersBatchResponse {
  total: number
  found: number
  missing: string[]
  papers: Array<{
    paper_id: string
    title?: string
    authors?: string[]
    abstract?: string
    categories?: string[]
    publish_date?: string
    chunk_count?: number
    has_images?: boolean
    minio_pdf_url?: string
    minio_markdown_url?: string
  }>
}

// Frontend response model (what the UI expects)
export interface SearchResponse {
  contextId: string
  papers: PaperResult[]
  total: number
  page: number
  pageSize: number
}

// --- QA Response Models ---
export interface BackendQAResponse {
  answer: string
  sources: Array<Record<string, any>>
  confidence_score: number
  processing_time: number
  context_chunks_count: number
  papers_involved: string[]
}

export interface QASource extends Record<string, any> {}

export interface QAResponse {
  answer: string
  sources: QASource[]
  confidenceScore: number
  processingTime: number
  contextChunksCount: number
  papersInvolved: string[]
}

// Note: QA, Summary, and other interfaces removed as they're not implemented in backend

export class ResearchService {
  // Helper function to transform backend paper data to frontend PaperResult
  private transformPaper(backendPaper: any): PaperResult {
    return {
      id: backendPaper.paper_id,
      title: backendPaper.title,
      authors: backendPaper.authors || [],
      abstract: backendPaper.abstract || "",
      citationCount: 0, // Not available in backend
      publicationDate: backendPaper.publish_date || "",
      venue: "", // Not available in backend
      doi: "", // Not available in backend
      url: backendPaper.minio_pdf_url || `https://arxiv.org/pdf/${backendPaper.paper_id}`, // Use PDF URL if available
      keywords: backendPaper.categories || [],
      pdfUrl: backendPaper.minio_pdf_url || `https://arxiv.org/pdf/${backendPaper.paper_id}`, // Use PDF URL if available
      isBookmarked: false,
      isOpenAccess: false, // Not available in backend
      impactFactor: backendPaper.score || 0,
      journalRank: "",
      // Pass through evidence from agent responses if present
      evidenceSentences: backendPaper.evidence_sentences || backendPaper.evidenceSentences || []
    }
  }

  /**
   * Map Backend QAResponse (snake_case) -> Frontend QAResponse (camelCase).
   * Chỉ chuyển đổi tên trường; giữ nguyên cấu trúc mảng sources từ backend.
   */
  private transformQAResponse(backend: BackendQAResponse): QAResponse {
    return {
      answer: backend.answer,
      sources: backend.sources || [],
      confidenceScore: backend.confidence_score ?? 0,
      processingTime: backend.processing_time ?? 0,
      contextChunksCount: backend.context_chunks_count ?? 0,
      papersInvolved: backend.papers_involved || [],
    }
  }

  async searchPapers(
    query: string,
    filters?: SearchFilters,
    page = 1,
    pageSize = 20,
    backendMode: 'fulltext' | 'hybrid' | 'semantic' = 'fulltext',
  ): Promise<SearchResponse> {
    // Map frontend parameters to backend SearchRequest model
    const searchRequest = {
      query,
      max_results: pageSize,
      search_mode: backendMode,
      // Only date and author currently supported in UI
      date_from: filters?.dateRange?.from,
      date_to: filters?.dateRange?.to,
      author: filters?.authors?.[0] // Take first author if multiple
    }

    const backendResponse = await apiFetch<BackendSearchResponse>(`/api/v1/search`, {
      method: 'POST',
      body: JSON.stringify(searchRequest),
    })

    // Transform backend response to frontend format
    return {
      contextId: `search_${Date.now()}`, // Generate a context ID
      papers: backendResponse.results.map((paper) => this.transformPaper(paper)),
      total: backendResponse.total,
      page: page,
      pageSize: pageSize
    }
  }

  async searchPapersWithAgent(query: string, _filters?: SearchFilters, page = 1, pageSize = 20): Promise<SearchResponse> {
    // Map to backend LlamaSearchRequest model
    const agentRequest = {
      query,
      max_results: pageSize,
      enable_iterations: true,
      include_summaries: true
    }

    const backendResponse = await apiFetch<BackendAgentResponse>(`/api/v1/llama/search`, {
      method: 'POST',
      body: JSON.stringify(agentRequest),
      timeoutMs: 300000, // Allow longer runtime for agent searches
    })

    // Handle agent-declared failures gracefully
    if (!backendResponse.success) {
      const err = new Error(backendResponse.error || 'Agent search failed') as Error & { code?: string }
      const msg = (backendResponse.error || '').toLowerCase()
      if (msg.includes('quality is too low') || msg.includes('no relevant papers')) {
        err.code = 'NO_RELEVANT_PAPERS'
      }
      throw err
    }

    // Transform backend agent response to frontend format
    return {
      contextId: `agent_search_${Date.now()}`, // Generate a context ID
      papers: backendResponse.papers.map((paper) => {
        const mapped = this.transformPaper(paper)
        // Attach evidence sentences if present
        ;(mapped as any).evidenceSentences = (paper as any).evidence_sentences || []
        return mapped
      }),
      total: backendResponse.total_found,
      page: page,
      pageSize: pageSize
    }
  }

  // Paper details API integration
  async getPaperDetails(paperId: string) {
    return apiFetch<{
      paper_id: string
      title: string
      authors: string[]
      abstract: string
      content: string
      content_length: number
      categories: string[]
      publish_date: string
      word_count: number
      chunk_count: number
      has_images: boolean
      pdf_size: number
      downloaded_at: string
      indexed_at: string
      markdown_path: string
      pdf_path: string
      minio_pdf_url: string
      minio_markdown_url: string
    }>(`/api/v1/papers/${encodeURIComponent(paperId)}`, { method: 'GET' })
  }

  async findSimilarPapers(paperId: string, maxResults = 10) {
    return apiFetch<{
      reference_paper_id: string
      similar_papers: Array<{
        paper_id: string
        title: string
        authors: string[]
        abstract: string
        similarity_score: number
        categories: string[]
        publish_date: string
      }>
      total: number
    }>(`/api/v1/papers/${encodeURIComponent(paperId)}/similar`, {
      method: 'POST',
      body: JSON.stringify({ max_results: maxResults })
    })
  }

  // NOTE: Bookmark functionality not implemented in backend yet
  async bookmark(paperId: string, on: boolean) {
    // For now, just return success - can be implemented when backend supports it
    console.warn('Bookmark functionality not yet implemented in backend')
    return Promise.resolve()
  }

  // --- Suggest ---
  async suggest(q: string, maxResults = 5) {
    return apiFetch<{ query: string; suggestions: string[] }>(`/api/v1/search/suggest?query=${encodeURIComponent(q)}&max_results=${maxResults}`, { method: 'GET' })
  }

  // Note: QA, Summary, and Documents endpoints are not implemented in the current backend
  // These would need to be implemented in the backend first before adding them here

  // --- QA (stubs) ---
  /**
   * Hỏi đáp theo danh sách paper đã chọn.
   * - 1 paper: gọi /api/v1/qa/single-paper { paper_id, question }
   * - >1 papers: gọi /api/v1/qa/multi-paper { paper_ids, question }
   */
  async qaSelected(params: { question: string; paperIds: string[]; citationStyle?: string; retrieval?: string; conversationId?: string }): Promise<QAResponse> {
    const ids = (params.paperIds || []).map((s) => s.trim()).filter(Boolean)
    if (!params.question || ids.length === 0) {
      throw new Error('Missing question or paperIds')
    }

    if (ids.length === 1) {
      const payload = { paper_id: ids[0], question: params.question }
      const resp = await apiFetch<BackendQAResponse>(`/api/v1/qa/single-paper`, {
        method: 'POST',
        body: JSON.stringify(payload),
        timeoutMs: 300000,
      })
      return this.transformQAResponse(resp)
    }

    const payload = { paper_ids: ids, question: params.question }
    const resp = await apiFetch<BackendQAResponse>(`/api/v1/qa/multi-paper`, {
      method: 'POST',
      body: JSON.stringify(payload),
      timeoutMs: 300000,
    })
    return this.transformQAResponse(resp)
  }

  /**
   * Hỏi đáp theo tập nhiều paper. Khuyến nghị truyền trực tiếp danh sách paperIds.
   * Lưu ý: Tham số searchContextId không được backend sử dụng cho QA hiện tại.
   */
  async qaAll(_params: { question: string; searchContextId: string; paperIds?: string[]; citationStyle?: string; retrieval?: string; conversationId?: string }): Promise<QAResponse> {
    const ids = (_params.paperIds || []).map((s) => s.trim()).filter(Boolean)
    if (!ids.length) {
      throw new Error('qaAll requires paperIds; use qaSelected or provide paperIds')
    }
    // Delegate sang multi-paper
    const payload = { paper_ids: ids, question: _params.question }
    const resp = await apiFetch<BackendQAResponse>(`/api/v1/qa/multi-paper`, {
      method: 'POST',
      body: JSON.stringify(payload),
      timeoutMs: 300000,
    })
    return this.transformQAResponse(resp)
  }

  /**
   * Fetch lightweight metadata for a batch of paper IDs.
   * Used when user selects papers (tick) in QA / Summary modes so we can build
   * a reliable context without requiring manual paper_id input.
   */
  async fetchPapersBatch(paperIds: string[]): Promise<{ papers: PaperResult[]; missing: string[] }> {
    if (!paperIds || paperIds.length === 0) {
      return { papers: [], missing: [] }
    }

    const unique = Array.from(new Set(paperIds.map((p) => p.trim()).filter(Boolean)))
    // Backend limit per request is 50 (enforced in model) – enforce here defensively
    const MAX_BATCH = 50
    const chunks: string[][] = []
    for (let i = 0; i < unique.length; i += MAX_BATCH) {
      chunks.push(unique.slice(i, i + MAX_BATCH))
    }

    const aggregate: PaperResult[] = []
    const missingAll: string[] = []

    for (const chunk of chunks) {
      try {
        const resp = await apiFetch<BackendPapersBatchResponse>(`/api/v1/papers/batch`, {
          method: 'POST',
          body: JSON.stringify({ paper_ids: chunk })
        })
        if (resp?.papers?.length) {
          aggregate.push(...resp.papers.map(p => this.transformPaper(p)))
        }
        if (resp?.missing?.length) missingAll.push(...resp.missing)
      } catch (e) {
        console.error('[fetchPapersBatch] error chunk', e)
      }
    }

    return { papers: aggregate, missing: missingAll }
  }
}