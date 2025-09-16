"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Bot, User, ChevronDown, Quote, ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import { useState } from "react"

export interface MessageWithCitations {
  id: string
  content: string
  role: "user" | "assistant"
  timestamp: Date
  sources?: Array<{
    id: string
    title: string
    authors: string[]
    url?: string
    type: "paper" | "document" | "url"
  }>
  bibliography?: string[]
}

interface ChatMessageProps {
  message: MessageWithCitations
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [showBibliography, setShowBibliography] = useState(false)

  const formatMessageContent = (content: string) => {
    // Convert citation markers to clickable elements
    return content.split(/(\[[0-9]+\])/).map((part, index) => {
      const citationMatch = part.match(/\[([0-9]+)\]/)
      if (citationMatch) {
        const citationNumber = Number.parseInt(citationMatch[1])
        const source = message.sources?.[citationNumber - 1]

        return (
          <button
            key={index}
            className="inline-flex items-center px-1 py-0.5 text-xs bg-primary/10 text-primary rounded hover:bg-primary/20 transition-colors"
            onClick={() => source?.url && window.open(source.url, "_blank")}
            title={source ? `${source.title} - ${source.authors.join(", ")}` : "Citation"}
          >
            [{citationNumber}]
          </button>
        )
      }
      return part
    })
  }

  return (
    <div className={cn("flex gap-3", message.role === "user" ? "justify-end" : "justify-start")}>
      {message.role === "assistant" && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
            <Bot className="h-4 w-4 text-primary-foreground" />
          </div>
        </div>
      )}

      <div className={cn("max-w-[80%] space-y-2", message.role === "user" ? "items-end" : "items-start")}>
        <Card
          className={cn(
            "p-4",
            message.role === "user" ? "bg-primary text-primary-foreground" : "bg-card text-card-foreground",
          )}
        >
          <div className="text-sm leading-relaxed text-pretty">
            {message.role === "assistant" ? formatMessageContent(message.content) : message.content}
          </div>
          <div
            className={cn(
              "text-xs mt-2 opacity-70",
              message.role === "user" ? "text-primary-foreground" : "text-muted-foreground",
            )}
          >
            {message.timestamp.toLocaleTimeString()}
          </div>
        </Card>

        {/* Sources and Bibliography for Assistant Messages */}
        {message.role === "assistant" && (message.sources?.length || message.bibliography?.length) && (
          <div className="w-full space-y-2">
            {/* Sources */}
            {message.sources && message.sources.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {message.sources.map((source, index) => (
                  <div key={source.id} className="flex items-center gap-1">
                    <Badge variant="outline" className="text-xs">
                      <Quote className="h-3 w-3 mr-1" />[{index + 1}] {source.type}
                    </Badge>
                    {source.url && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={() => window.open(source.url, "_blank")}
                      >
                        <ExternalLink className="h-3 w-3" />
                      </Button>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Bibliography */}
            {message.bibliography && message.bibliography.length > 0 && (
              <Collapsible open={showBibliography} onOpenChange={setShowBibliography}>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-8 text-xs">
                    <ChevronDown
                      className={cn("h-3 w-3 mr-1 transition-transform", showBibliography && "rotate-180")}
                    />
                    References ({message.bibliography.length})
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <Card className="p-3 mt-2 bg-muted/50">
                    <h4 className="text-xs font-semibold mb-2 text-muted-foreground uppercase tracking-wide">
                      References
                    </h4>
                    <div className="space-y-2">
                      {message.bibliography.map((citation, index) => (
                        <p key={index} className="text-xs leading-relaxed text-muted-foreground">
                          [{index + 1}] {citation}
                        </p>
                      ))}
                    </div>
                  </Card>
                </CollapsibleContent>
              </Collapsible>
            )}
          </div>
        )}
      </div>

      {message.role === "user" && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
            <User className="h-4 w-4 text-secondary-foreground" />
          </div>
        </div>
      )}
    </div>
  )
}
