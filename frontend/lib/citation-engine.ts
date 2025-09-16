export interface CitationSource {
    id: string
    type: "paper" | "document" | "url"
    title: string
    authors: string[]
    year?: string
    venue?: string
    url?: string
    doi?: string
    pages?: string
    abstract?: string
  }
  
  export interface CitationStyle {
    name: string
    format: (source: CitationSource, index?: number) => string
    inTextFormat: (source: CitationSource, index: number) => string
  }
  
  export class CitationEngine {
    private sources: Map<string, CitationSource> = new Map()
    private citationCounter = 0
  
    // Citation styles
    static styles: Record<string, CitationStyle> = {
      apa: {
        name: "APA",
        format: (source: CitationSource) => {
          const authors =
            source.authors.length > 0
              ? source.authors.length === 1
                ? source.authors[0]
                : source.authors.length === 2
                  ? `${source.authors[0]} & ${source.authors[1]}`
                  : `${source.authors[0]} et al.`
              : "Unknown Author"
  
          const year = source.year ? `(${source.year})` : "(n.d.)"
          const venue = source.venue ? `. ${source.venue}` : ""
          const doi = source.doi ? `. https://doi.org/${source.doi}` : ""
          const url = !source.doi && source.url ? `. ${source.url}` : ""
  
          return `${authors} ${year}. ${source.title}${venue}${doi}${url}`
        },
        inTextFormat: (source: CitationSource, index: number) => {
          const authors =
            source.authors.length > 0
              ? source.authors.length === 1
                ? source.authors[0].split(" ").pop() || source.authors[0]
                : source.authors.length === 2
                  ? `${source.authors[0].split(" ").pop()} & ${source.authors[1].split(" ").pop()}`
                  : `${source.authors[0].split(" ").pop()} et al.`
              : "Unknown"
  
          const year = source.year || "n.d."
          return `(${authors}, ${year})`
        },
      },
      ieee: {
        name: "IEEE",
        format: (source: CitationSource, index?: number) => {
          const authors = source.authors.length > 0 ? source.authors.join(", ") : "Unknown Author"
  
          const year = source.year ? `, ${source.year}` : ""
          const venue = source.venue ? `, ${source.venue}` : ""
          const doi = source.doi ? `, doi: ${source.doi}` : ""
          const url = !source.doi && source.url ? `, [Online]. Available: ${source.url}` : ""
  
          return `${authors}, "${source.title}"${venue}${year}${doi}${url}`
        },
        inTextFormat: (source: CitationSource, index: number) => `[${index}]`,
      },
      mla: {
        name: "MLA",
        format: (source: CitationSource) => {
          const authors =
            source.authors.length > 0
              ? source.authors.length === 1
                ? source.authors[0]
                : source.authors.length === 2
                  ? `${source.authors[0]} and ${source.authors[1]}`
                  : `${source.authors[0]} et al.`
              : "Unknown Author"
  
          const venue = source.venue ? ` ${source.venue},` : ""
          const year = source.year ? ` ${source.year}` : ""
          const url = source.url ? ` Web. ${new Date().toLocaleDateString()}.` : ""
  
          return `${authors}. "${source.title}."${venue}${year}.${url}`
        },
        inTextFormat: (source: CitationSource, index: number) => {
          const author = source.authors.length > 0 ? source.authors[0].split(" ").pop() || source.authors[0] : "Unknown"
          return `(${author})`
        },
      },
    }
  
    addSource(source: CitationSource): string {
      if (!this.sources.has(source.id)) {
        this.sources.set(source.id, source)
        this.citationCounter++
      }
      return source.id
    }
  
    getSource(id: string): CitationSource | undefined {
      return this.sources.get(id)
    }
  
    getAllSources(): CitationSource[] {
      return Array.from(this.sources.values())
    }
  
    generateInTextCitation(sourceId: string, style = "apa"): string {
      const source = this.sources.get(sourceId)
      if (!source) return "[Citation not found]"
  
      const citationStyle = CitationEngine.styles[style]
      if (!citationStyle) return "[Invalid citation style]"
  
      const index = Array.from(this.sources.keys()).indexOf(sourceId) + 1
      return citationStyle.inTextFormat(source, index)
    }
  
    generateBibliography(style = "apa"): string[] {
      const citationStyle = CitationEngine.styles[style]
      if (!citationStyle) return ["Invalid citation style"]
  
      return Array.from(this.sources.values()).map((source, index) => citationStyle.format(source, index + 1))
    }
  
    formatTextWithCitations(text: string, sourceIds: string[], style = "apa"): string {
      let formattedText = text
  
      sourceIds.forEach((sourceId, index) => {
        const citation = this.generateInTextCitation(sourceId, style)
        const placeholder = `[${index + 1}]`
        formattedText = formattedText.replace(placeholder, citation)
      })
  
      return formattedText
    }
  
    clear(): void {
      this.sources.clear()
      this.citationCounter = 0
    }
  }