"use client"

import { Card } from "@/components/ui/card"
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
}

interface ChatMessageProps {
  message: MessageWithCitations
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
                {message.content}
              </ReactMarkdown>
            ) : (
              message.content
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
