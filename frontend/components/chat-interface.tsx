"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Send, Bot, Upload, MessageSquare, Settings } from "lucide-react"
import { PDFUploadInterface, type UploadedDocument } from "@/components/pdf-upload-interface"
import { ChatMessage, type MessageWithCitations } from "@/components/chat-message"

interface ChatInterfaceProps {
  messages: MessageWithCitations[]
  onSendMessage: (content: string) => void
  documents?: UploadedDocument[]
  onDocumentAdd?: (document: UploadedDocument) => void
  onDocumentRemove?: (documentId: string) => void
  onDocumentSelect?: (documentId: string) => void
  selectedDocumentId?: string
  citationStyle?: string
  onCitationStyleChange?: (style: string) => void
  isLoading?: boolean
}

export function ChatInterface({
  messages,
  onSendMessage,
  documents = [],
  onDocumentAdd,
  onDocumentRemove,
  onDocumentSelect,
  selectedDocumentId,
  citationStyle = "apa",
  onCitationStyleChange,
  isLoading = false,
}: ChatInterfaceProps) {
  const [input, setInput] = useState("")
  const [activeTab, setActiveTab] = useState<"chat" | "upload">("chat")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim())
      setInput("")
    }
  }

  const handleDocumentAdd = (document: UploadedDocument) => {
    onDocumentAdd?.(document)
  }

  const handleDocumentRemove = (documentId: string) => {
    onDocumentRemove?.(documentId)
  }

  const handleDocumentSelect = (documentId: string) => {
    onDocumentSelect?.(documentId)
    setActiveTab("chat")
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="hidden lg:block p-6 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground text-balance">Academic Paper Search & Summarization</h1>
            <p className="text-muted-foreground mt-2 text-pretty">
              Search for research papers, get summaries, and explore academic literature with AI assistance.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Settings className="h-4 w-4 text-muted-foreground" />
            <Select value={citationStyle} onValueChange={onCitationStyleChange}>
              <SelectTrigger className="w-24">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="apa">APA</SelectItem>
                <SelectItem value="ieee">IEEE</SelectItem>
                <SelectItem value="mla">MLA</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      <Tabs
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as "chat" | "upload")}
        className="flex-1 flex flex-col"
      >
        <div className="border-b border-border bg-card px-4">
          <TabsList className="h-12 w-full justify-start rounded-none bg-transparent p-0">
            <TabsTrigger
              value="chat"
              className="flex items-center gap-2 h-12 rounded-none border-b-2 border-transparent data-[state=active]:border-primary"
            >
              <MessageSquare className="h-4 w-4" />
              Chat
              {messages.length > 0 && (
                <span className="ml-1 px-2 py-0.5 text-xs bg-primary/10 text-primary rounded-full">
                  {messages.length}
                </span>
              )}
            </TabsTrigger>
            <TabsTrigger
              value="upload"
              className="flex items-center gap-2 h-12 rounded-none border-b-2 border-transparent data-[state=active]:border-primary"
            >
              <Upload className="h-4 w-4" />
              Documents
              {documents.length > 0 && (
                <span className="ml-1 px-2 py-0.5 text-xs bg-secondary/10 text-secondary rounded-full">
                  {documents.length}
                </span>
              )}
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="chat" className="flex-1 flex flex-col mt-0">
          {selectedDocumentId && (
            <div className="px-4 py-2 bg-muted/50 border-b border-border">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-muted-foreground">
                    Chatting with: {documents.find((d) => d.id === selectedDocumentId)?.name || "Selected document"}
                  </span>
                </div>
                <div className="lg:hidden">
                  <Select value={citationStyle} onValueChange={onCitationStyleChange}>
                    <SelectTrigger className="w-20 h-6 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="apa">APA</SelectItem>
                      <SelectItem value="ieee">IEEE</SelectItem>
                      <SelectItem value="mla">MLA</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          )}

          {/* Messages */}
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4 max-w-4xl mx-auto">
              {messages.length === 0 ? (
                <div className="text-center py-12">
                  <Bot className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-foreground mb-2">Start Your Research</h3>
                  <p className="text-muted-foreground text-pretty">
                    {documents.length > 0
                      ? "Select a document from the Documents tab and start asking questions about it."
                      : "Upload documents or ask me to search for papers, summarize research, or explore specific topics in academic literature."}
                  </p>
                </div>
              ) : (
                messages.map((message) => <ChatMessage key={message.id} message={message} />)
              )}

              {isLoading && (
                <div className="flex gap-3 justify-start">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                      <Bot className="h-4 w-4 text-primary-foreground" />
                    </div>
                  </div>
                  <div className="bg-card text-card-foreground p-4 rounded-lg border">
                    <div className="flex items-center gap-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                      <span className="text-sm text-muted-foreground">Generating response with citations...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Input */}
          <div className="p-4 border-t border-border">
            <form onSubmit={handleSubmit} className="flex gap-2 max-w-4xl mx-auto">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  selectedDocumentId
                    ? "Ask questions about the selected document..."
                    : "Search for papers, ask for summaries, or explore research topics..."
                }
                className="flex-1"
                disabled={isLoading}
              />
              <Button type="submit" disabled={!input.trim() || isLoading}>
                <Send className="h-4 w-4" />
              </Button>
            </form>
          </div>
        </TabsContent>

        <TabsContent value="upload" className="flex-1 mt-0">
          <ScrollArea className="h-full">
            <div className="p-6">
              <PDFUploadInterface
                documents={documents}
                onDocumentAdd={handleDocumentAdd}
                onDocumentRemove={handleDocumentRemove}
                onDocumentSelect={handleDocumentSelect}
                selectedDocumentId={selectedDocumentId}
              />
            </div>
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  )
}
