"use client"

import { Card } from "@/components/ui/card"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Bot, User } from "lucide-react"
import { cn } from "@/lib/utils"
import { useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

export interface MessageWithCitations {
  id: string
  content: string
  role: "user" | "assistant"
  timestamp: Date
  // citations and bibliography removed from UI rendering per spec
  sources?: Array<unknown>
  bibliography?: string[]
  sections?: Array<{
    title?: string
    content: string
    summary?: string
  }>
}

interface ChatMessageProps {
  message: MessageWithCitations
}

function normalizeMarkdownText(text: string): string {
  return String(text || "")
    // xuống dòng trước dấu "-"
    .replace(/(?:^|\s)-\s+/g, "\n- ")
    // xuống dòng trước dấu "*"
    .replace(/(?:^|\s)\*\s+/g, "\n* ")
    // nếu có ": -" thì cũng xuống dòng
    .replace(/:\s*-\s+/g, ":\n- ")
    // ép xuống dòng sau dấu chấm + space + chữ hoa
    .replace(/\. ([A-Z])/g, ".\n$1")
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [/* showBibliography */] = useState(false)

  return (
    <div className={cn("flex gap-3", message.role === "user" ? "justify-end" : "justify-start")}>
      {message.role === "assistant" && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center">
            <Bot className="h-4 w-4 text-white" />
          </div>
        </div>
      )}

      <div className={cn("max-w-[80%] space-y-2", message.role === "user" ? "items-end" : "items-start")}>
        <Card
          className={cn(
            "p-4",
            message.role === "user"
              ? "bg-emerald-600 text-white"
              : "bg-white text-slate-800 border border-green-200",
          )}
        >
          <div className="text-sm leading-relaxed text-pretty">
            {message.role === "assistant" ? (
              message.sections && message.sections.length > 0 ? (
                <Accordion type="multiple" className="w-full">
                  {message.sections.map((sec, idx) => (
                    <AccordionItem key={idx} value={`sec-${idx}`}>
                      <AccordionTrigger className="text-foreground">
                        <div className="flex flex-col items-start text-left">
                          <span className="font-semibold">
                            {sec.title || `Section ${idx + 1}`}
                          </span>
                          {sec.summary && (
                            <span className="text-xs text-muted-foreground mt-1">
                              {sec.summary}
                            </span>
                          )}
                        </div>
                      </AccordionTrigger>
                      <AccordionContent>
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            a: ({ node, ...props }) => (
                              <a {...props} className="underline text-emerald-700" target="_blank" rel="noreferrer" />
                            ),
                            code: ((props: any) => {
                              const { inline, className, children, ...rest } = props || {}
                              return (
                              <code
                                {...rest}
                                className={cn(
                                  "rounded px-1.5 py-0.5",
                                  inline ? "bg-slate-100 text-slate-800" : "block bg-slate-900 text-slate-100 p-3 overflow-auto"
                                )}
                              >
                                {children}
                              </code>
                              )
                            }),
                            h1: ({ children }) => <h1 className="text-lg font-semibold mb-2">{children}</h1>,
                            h2: ({ children }) => <h2 className="text-base font-semibold mb-2">{children}</h2>,
                            ul: ({ children }) => <ul className="list-disc pl-5 space-y-1">{children}</ul>,
                            ol: ({ children }) => <ol className="list-decimal pl-5 space-y-1">{children}</ol>,
                          }}
                        >
                          {normalizeMarkdownText(sec.content)}
                        </ReactMarkdown>
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              ) : (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    a: ({ node, ...props }) => (
                      <a {...props} className="underline text-emerald-700" target="_blank" rel="noreferrer" />
                    ),
                    code: ((props: any) => {
                      const { inline, className, children, ...rest } = props || {}
                      return (
                      <code
                        {...rest}
                        className={cn(
                          "rounded px-1.5 py-0.5",
                          inline ? "bg-slate-100 text-slate-800" : "block bg-slate-900 text-slate-100 p-3 overflow-auto"
                        )}
                      >
                        {children}
                      </code>
                      )
                    }),
                    h1: ({ children }) => <h1 className="text-lg font-semibold mb-2">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-base font-semibold mb-2">{children}</h2>,
                    ul: ({ children }) => <ul className="list-disc pl-5 space-y-1">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal pl-5 space-y-1">{children}</ol>,
                  }}
                >
                  {normalizeMarkdownText(message.content)}
                </ReactMarkdown>
              )
            ) : (
              normalizeMarkdownText(message.content)
            )}
          </div>
          {message.role === "user" && (
            <div
              className={cn(
                "text-xs mt-2 opacity-70",
                "text-white/80",
              )}
            >
              {message.timestamp.toLocaleTimeString()}
            </div>
          )}
        </Card>

      </div>

      {message.role === "user" && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center">
            <User className="h-4 w-4 text-emerald-700" />
          </div>
        </div>
      )}
    </div>
  )
}
