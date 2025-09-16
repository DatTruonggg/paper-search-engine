"use client"

import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ExternalLink, FileText, Users, Calendar } from "lucide-react"
import type { Paper } from "@/app/page"

interface PaperResultsProps {
  papers: Paper[]
}

export function PaperResults({ papers }: PaperResultsProps) {
  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-border">
        <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
          <FileText className="h-5 w-5" />
          Research Papers
        </h2>
        <p className="text-sm text-muted-foreground mt-1">{papers.length} papers found</p>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {papers.length === 0 ? (
            <div className="text-center py-8">
              <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground text-pretty">
                No papers found yet. Start a conversation to search for research papers.
              </p>
            </div>
          ) : (
            papers.map((paper) => (
              <Card key={paper.id} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium leading-tight text-balance">{paper.title}</CardTitle>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Users className="h-3 w-3" />
                    <span className="truncate">
                      {paper.authors.slice(0, 2).join(", ")}
                      {paper.authors.length > 2 && ` +${paper.authors.length - 2} more`}
                    </span>
                  </div>
                </CardHeader>

                <CardContent className="pt-0 space-y-3">
                  <p className="text-xs text-muted-foreground leading-relaxed text-pretty">
                    {paper.abstract.length > 150 ? `${paper.abstract.substring(0, 150)}...` : paper.abstract}
                  </p>

                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Calendar className="h-3 w-3" />
                    <span>{new Date(paper.publishedDate).getFullYear()}</span>
                    {paper.citationCount && (
                      <>
                        <span>•</span>
                        <Badge variant="secondary" className="text-xs">
                          {paper.citationCount.toLocaleString()} citations
                        </Badge>
                      </>
                    )}
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full justify-center gap-2 bg-transparent"
                    onClick={() => window.open(paper.url, "_blank")}
                  >
                    <ExternalLink className="h-3 w-3" />
                    View Paper
                  </Button>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
