import { CitationEngine, type CitationSource } from "./citation-engine"
import type { UploadedDocument } from "@/components/pdf-upload-interface"
import type { PaperMetadata } from "@/components/paper-search-engine"

export interface ChatContext {
  documents: UploadedDocument[]
  papers: PaperMetadata[]
  selectedDocumentId?: string
  citationStyle: string
}

export interface ChatResponse {
  content: string
  sources: CitationSource[]
  bibliography?: string[]
}

export class ChatService {
  private citationEngine: CitationEngine

  constructor() {
    this.citationEngine = new CitationEngine()
  }

  async generateResponse(query: string, context: ChatContext): Promise<ChatResponse> {
    // Clear previous citations for new response
    this.citationEngine.clear()

    // Convert documents and papers to citation sources
    const availableSources: CitationSource[] = [
      ...context.documents
        .filter((doc) => doc.status === "ready")
        .map((doc) => ({
          id: doc.id,
          type: "document" as const,
          title: doc.metadata?.title || doc.name,
          authors: doc.metadata?.authors || ["Unknown Author"],
          year: doc.metadata ? new Date().getFullYear().toString() : undefined,
          url: doc.type === "url" ? doc.source : undefined,
          abstract: doc.metadata?.abstract,
        })),
      ...context.papers.map((paper) => ({
        id: paper.id,
        type: "paper" as const,
        title: paper.title,
        authors: paper.authors,
        year: new Date(paper.publishedDate).getFullYear().toString(),
        venue: paper.venue,
        url: paper.url,
        doi: paper.doi,
        abstract: paper.abstract,
      })),
    ]

    // Simulate AI processing and response generation
    const response = await this.processQuery(query, availableSources, context)

    return response
  }

  private async processQuery(query: string, sources: CitationSource[], context: ChatContext): Promise<ChatResponse> {
    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 1000))

    let responseContent = ""
    const usedSources: CitationSource[] = []

    // Determine response type based on query
    if (query.toLowerCase().includes("summary") || query.toLowerCase().includes("summarize")) {
      responseContent = this.generateSummaryResponse(query, sources, context)
      usedSources.push(...sources.slice(0, Math.min(3, sources.length)))
    } else if (query.toLowerCase().includes("compare")) {
      responseContent = this.generateComparisonResponse(query, sources, context)
      usedSources.push(...sources.slice(0, Math.min(2, sources.length)))
    } else if (query.toLowerCase().includes("methodology") || query.toLowerCase().includes("method")) {
      responseContent = this.generateMethodologyResponse(query, sources, context)
      usedSources.push(...sources.slice(0, Math.min(2, sources.length)))
    } else {
      responseContent = this.generateGeneralResponse(query, sources, context)
      usedSources.push(...sources.slice(0, Math.min(2, sources.length)))
    }

    // Add sources to citation engine
    usedSources.forEach((source) => this.citationEngine.addSource(source))

    // Generate bibliography
    const bibliography = this.citationEngine.generateBibliography(context.citationStyle)

    return {
      content: responseContent,
      sources: usedSources,
      bibliography,
    }
  }

  private generateSummaryResponse(query: string, sources: CitationSource[], context: ChatContext): string {
    if (context.selectedDocumentId) {
      const selectedSource = sources.find((s) => s.id === context.selectedDocumentId)
      if (selectedSource) {
        return `Based on "${selectedSource.title}" [1], here are the key findings:\n\n• The paper presents novel approaches to the research problem with significant implications for the field.\n• The methodology employed demonstrates rigorous experimental design and validation.\n• The results show measurable improvements over existing approaches, with statistical significance.\n• The authors conclude that their findings contribute meaningfully to current understanding.\n\nThe research methodology is particularly noteworthy for its comprehensive approach [1]. The implications of these findings extend beyond the immediate scope of the study.`
      }
    }

    if (sources.length > 0) {
      return `Based on the available literature [1], here's a comprehensive summary:\n\n• The research demonstrates significant advances in the field with practical applications.\n• Multiple studies confirm the effectiveness of the proposed approaches [1].\n• The methodological frameworks presented offer robust solutions to existing challenges.\n• Future research directions are clearly outlined with specific recommendations.\n\nThese findings collectively suggest a promising direction for continued investigation in this area.`
    }

    return "I'd be happy to provide a summary, but I don't see any documents or papers available in the current session. Please upload documents or search for papers first."
  }

  private generateComparisonResponse(query: string, sources: CitationSource[], context: ChatContext): string {
    if (sources.length >= 2) {
      return `Comparing the approaches presented in the literature [1][2]:\n\n**Methodological Differences:**\n• The first study [1] employs a quantitative approach with large-scale data analysis.\n• The second work [2] focuses on qualitative methods with in-depth case studies.\n\n**Key Findings:**\n• Both studies confirm the importance of the research problem [1][2].\n• However, they differ in their proposed solutions and implementation strategies.\n• The effectiveness metrics vary, with [1] emphasizing statistical significance and [2] focusing on practical applicability.\n\n**Implications:**\nThe complementary nature of these approaches suggests that a hybrid methodology might yield optimal results.`
    }

    return `To provide a meaningful comparison, I need at least two sources. Currently, I have access to ${sources.length} source(s). Please add more documents or papers to enable comparative analysis.`
  }

  private generateMethodologyResponse(query: string, sources: CitationSource[], context: ChatContext): string {
    if (sources.length > 0) {
      return `The methodology described in the literature [1] follows a systematic approach:\n\n**Research Design:**\n• The study employs a mixed-methods approach combining quantitative and qualitative techniques [1].\n• Data collection involves multiple phases with rigorous validation procedures.\n• The sample size and selection criteria are clearly defined and justified.\n\n**Implementation:**\n• The experimental setup follows established protocols with appropriate controls [1].\n• Statistical analysis methods are appropriate for the data type and research questions.\n• Ethical considerations are addressed throughout the research process.\n\n**Validation:**\n• Results are validated through multiple independent measures and cross-validation techniques.\n• The methodology demonstrates reproducibility and reliability [1].`
    }

    return "I'd be happy to discuss methodology, but I need access to relevant papers or documents. Please upload documents or search for papers that contain methodological information."
  }

  private generateGeneralResponse(query: string, sources: CitationSource[], context: ChatContext): string {
    if (context.selectedDocumentId) {
      const selectedSource = sources.find((s) => s.id === context.selectedDocumentId)
      if (selectedSource) {
        return `According to "${selectedSource.title}" [1], the research addresses your question through several key points:\n\n• The paper provides comprehensive coverage of the topic with detailed analysis.\n• The findings are supported by robust evidence and peer review [1].\n• The implications extend to both theoretical understanding and practical applications.\n• The authors suggest several avenues for future research and development.\n\nThe work contributes significantly to our understanding of the field and offers valuable insights for practitioners and researchers alike [1].`
      }
    }

    if (sources.length > 0) {
      return `Based on the available research [1], I can provide the following insights:\n\n• The literature demonstrates strong evidence supporting the main concepts in your query.\n• Current research trends indicate growing interest and development in this area [1].\n• The practical applications are well-documented with measurable outcomes.\n• Expert consensus supports the validity and reliability of the findings.\n\nThese sources provide a solid foundation for understanding the topic and its implications.`
    }

    return "I'd be happy to help answer your question. To provide the most accurate and well-cited response, please upload relevant documents or search for papers related to your query."
  }

  getCitationEngine(): CitationEngine {
    return this.citationEngine
  }
}
