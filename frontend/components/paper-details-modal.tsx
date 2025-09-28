"use client"

import { useState, useEffect } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  ExternalLink,
  Calendar,
  Users,
  Quote,
  BookOpen,
  Download,
  Star,
  StarOff,
  Copy,
  Award,
  Unlock,
  Eye,
  Lightbulb,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useToast } from "@/hooks/use-toast"
import type { PaperResult } from "./paper-results-panel"

interface PaperDetailsModalProps {
  paper: PaperResult | null
  isOpen: boolean
  onClose: () => void
  onBookmarkToggle?: (paperId: string) => void
  onViewFullPaper?: (paper: PaperResult) => void
}

export function PaperDetailsModal({
  paper,
  isOpen,
  onClose,
  onBookmarkToggle,
  onViewFullPaper,
}: PaperDetailsModalProps) {
  const { toast } = useToast()
  const [isBookmarked, setIsBookmarked] = useState(false)

  // Update local bookmark state when paper changes
  useEffect(() => {
    if (paper) {
      setIsBookmarked(paper.isBookmarked || false)
    }
  }, [paper])

  if (!paper) return null

  const hasEvidence = (paper as any).evidenceSentences && (paper as any).evidenceSentences.length > 0

  const handleBookmarkClick = async () => {
    // Optimistically update the local state immediately
    setIsBookmarked(!isBookmarked)

    // Call the parent handler
    if (onBookmarkToggle) {
      onBookmarkToggle(paper.id)
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    })
  }

  const formatCitationCount = (count: number) => {
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}k`
    }
    return count.toString()
  }

  const generateCitation = (paper: PaperResult) => {
    const authors = paper.authors.join(', ')
    const year = new Date(paper.publicationDate).getFullYear()
    return `${authors} (${year}). ${paper.title}. ${paper.venue || 'Journal'}. ${paper.doi ? `https://doi.org/${paper.doi}` : paper.url}`
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      toast({
        title: "Citation copied",
        description: "The citation has been copied to your clipboard.",
        duration: 3000,
      })
    }).catch(() => {
      toast({
        title: "Copy failed",
        description: "Unable to copy citation to clipboard.",
        variant: "destructive",
        duration: 3000,
      })
    })
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="w-[96vw] max-w-9xl h-[92vh] max-h-[92vh] p-0 flex flex-col">
        <DialogHeader className="px-4 sm:px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-green-50 to-emerald-50 flex-shrink-0">
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-3 mb-2">
                <DialogTitle className="text-lg sm:text-xl font-bold text-slate-800 leading-tight flex-1">
                  {paper.title}
                </DialogTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleBookmarkClick}
                  className="h-8 w-8 sm:h-9 sm:w-9 p-0 flex-shrink-0 ml-4"
                  title={isBookmarked ? "Remove bookmark" : "Add bookmark"}
                >
                  {isBookmarked ? (
                    <Star className="h-4 w-4 sm:h-5 sm:w-5 fill-current text-green-600" />
                  ) : (
                    <StarOff className="h-4 w-4 sm:h-5 sm:w-5 text-slate-400 hover:text-green-500" />
                  )}
                </Button>
              </div>
              <div className="flex items-center gap-2 sm:gap-4 text-xs sm:text-sm text-slate-600 flex-wrap">
                <div className="flex items-center gap-1">
                  <Users className="h-3 w-3 sm:h-4 sm:w-4" />
                  <span className="line-clamp-1">
                    {paper.authors.slice(0, 2).join(", ")}
                    {paper.authors.length > 2 && ` +${paper.authors.length - 2} more`}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <Calendar className="h-3 w-3 sm:h-4 sm:w-4" />
                  <span>{formatDate(paper.publicationDate)}</span>
                </div>
              </div>
            </div>
          </div>
        </DialogHeader>

        <div className="flex-1 min-h-0 px-4 sm:px-6 py-4 overflow-y-auto">
          <div className="space-y-4 sm:space-y-6">
            {/* Metadata Section */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              <div className="flex items-center gap-2 p-2 sm:p-3 bg-slate-50 rounded-lg">
                <Quote className="h-4 w-4 sm:h-5 sm:w-5 text-green-600" />
                <div>
                  <div className="text-xs text-slate-600">Citations</div>
                  <div className={cn(
                    "font-bold text-sm",
                    paper.citationCount > 100 ? "text-green-700" : "text-slate-700"
                  )}>
                    {formatCitationCount(paper.citationCount)}
                  </div>
                </div>
              </div>

              {paper.impactFactor && (
                <div className="flex items-center gap-2 p-2 sm:p-3 bg-slate-50 rounded-lg">
                  <Award className="h-4 w-4 sm:h-5 sm:w-5 text-amber-600" />
                  <div>
                    <div className="text-xs text-slate-600">Impact Factor</div>
                    <div className="font-bold text-sm text-slate-700">{paper.impactFactor}</div>
                  </div>
                </div>
              )}

              {paper.isOpenAccess && (
                <div className="flex items-center gap-2 p-2 sm:p-3 bg-green-50 rounded-lg border border-green-200">
                  <Unlock className="h-4 w-4 sm:h-5 sm:w-5 text-green-600" />
                  <div>
                    <div className="text-xs text-green-600">Access</div>
                    <div className="font-bold text-sm text-green-700">Open Access</div>
                  </div>
                </div>
              )}
            </div>

            {/* Venue */}
            {paper.venue && (
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
                  <BookOpen className="h-4 w-4" />
                  Published In
                </h3>
                <Badge className="text-sm bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium px-3 py-1">
                  {paper.venue}
                  {paper.journalRank && (
                    <span className="ml-2 text-xs text-green-700 font-bold">({paper.journalRank})</span>
                  )}
                </Badge>
              </div>
            )}

            {/* Keywords */}
            {paper.keywords && paper.keywords.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-3">Keywords</h3>
                <div className="flex flex-wrap gap-2">
                  {paper.keywords.map((keyword, index) => (
                    <Badge
                      key={index}
                      variant="outline"
                      className="text-xs px-2 py-1 border-green-200 text-green-700 hover:bg-green-50"
                    >
                      {keyword}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Abstract */}
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-2">Abstract</h3>
              <div className="p-3 sm:p-4 bg-slate-50 rounded-lg border border-slate-200">
                <p className="text-xs sm:text-sm text-slate-700 leading-relaxed text-justify">{paper.abstract}</p>
              </div>
            </div>

            {/* Evidence Section */}
            {hasEvidence && (
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <Lightbulb className="h-5 w-5 text-amber-500" />
                  <h3 className="text-lg font-semibold text-slate-800">AI-Extracted Evidence</h3>
                  <Badge className="text-xs bg-amber-50 text-amber-700 border border-amber-200">
                    {(paper as any).evidenceSentences.length} sentence{(paper as any).evidenceSentences.length !== 1 ? 's' : ''}
                  </Badge>
                </div>
                <div className="space-y-2 sm:space-y-3 max-h-48 sm:max-h-64 overflow-y-auto pr-1 sm:pr-2">
                  {((paper as any).evidenceSentences as string[]).map((sentence, idx) => (
                    <div key={idx} className="group">
                      <div className="relative p-3 sm:p-4 bg-gradient-to-r from-amber-50 to-yellow-50 border border-amber-200 rounded-lg hover:shadow-md transition-all duration-200">
                        <div className="absolute top-2 left-2 text-amber-400 text-lg sm:text-xl leading-none font-serif">"</div>
                        <div className="absolute bottom-2 right-2 text-amber-400 text-lg sm:text-xl leading-none font-serif rotate-180">"</div>
                        <blockquote className="px-4 sm:px-6 py-1 sm:py-2">
                          <p className="text-xs sm:text-sm text-slate-700 leading-relaxed italic font-medium">
                            {sentence}
                          </p>
                        </blockquote>
                        <div className="mt-2 pt-2 border-t border-amber-200">
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-amber-600 font-medium">Evidence #{idx + 1}</span>
                            <div className="flex items-center gap-1 text-xs text-amber-600">
                              <Lightbulb className="h-3 w-3" />
                              <span className="hidden sm:inline">AI Extracted</span>
                              <span className="sm:hidden">AI</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* DOI */}
            {paper.doi && (
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-2">DOI</h3>
                <code className="text-xs bg-slate-100 px-2 py-1 rounded border text-slate-600">
                  {paper.doi}
                </code>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="px-4 sm:px-6 py-3 sm:py-4 border-t border-slate-200 bg-slate-50 flex-shrink-0">
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2 sm:gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => copyToClipboard(generateCitation(paper))}
              className="flex items-center justify-center gap-2 text-xs sm:text-sm"
            >
              <Copy className="h-3 w-3 sm:h-4 sm:w-4" />
              <span className="hidden sm:inline">Copy Citation</span>
              <span className="sm:hidden">Copy</span>
            </Button>

            <div className="flex items-center gap-1 sm:gap-2">
              {paper.pdfUrl && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.open(paper.pdfUrl, "_blank")}
                  className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm flex-1 sm:flex-none justify-center"
                >
                  <Download className="h-3 w-3 sm:h-4 sm:w-4" />
                  <span className="hidden sm:inline">Download PDF</span>
                  <span className="sm:hidden">PDF</span>
                </Button>
              )}

              <Button
                variant="outline"
                size="sm"
                onClick={() => onViewFullPaper?.(paper)}
                className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm flex-1 sm:flex-none justify-center"
              >
                <Eye className="h-3 w-3 sm:h-4 sm:w-4" />
                <span className="hidden sm:inline">View Paper</span>
                <span className="sm:hidden">View</span>
              </Button>

              <Button
                size="sm"
                onClick={() => window.open(paper.url, "_blank")}
                className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white flex items-center gap-1 sm:gap-2 text-xs sm:text-sm flex-1 sm:flex-none justify-center"
              >
                <ExternalLink className="h-3 w-3 sm:h-4 sm:w-4" />
                <span className="sm:hidden">Open</span>
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}