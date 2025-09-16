"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, Link, FileText, X, CheckCircle, AlertCircle, Loader2 } from "lucide-react"

export interface UploadedDocument {
  id: string
  name: string
  type: "file" | "url"
  source: string
  status: "uploading" | "processing" | "ready" | "error"
  progress: number
  size?: number
  pages?: number
  error?: string
  extractedText?: string
  metadata?: {
    title?: string
    authors?: string[]
    abstract?: string
  }
}

interface PDFUploadInterfaceProps {
  documents: UploadedDocument[]
  onDocumentAdd: (document: UploadedDocument) => void
  onDocumentRemove: (documentId: string) => void
  onDocumentSelect: (documentId: string) => void
  selectedDocumentId?: string
}

export function PDFUploadInterface({
  documents,
  onDocumentAdd,
  onDocumentRemove,
  onDocumentSelect,
  selectedDocumentId,
}: PDFUploadInterfaceProps) {
  const [pdfUrl, setPdfUrl] = useState("")
  const [isProcessingUrl, setIsProcessingUrl] = useState(false)

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      acceptedFiles.forEach((file) => {
        if (file.type === "application/pdf") {
          const newDocument: UploadedDocument = {
            id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
            name: file.name,
            type: "file",
            source: file.name,
            status: "uploading",
            progress: 0,
            size: file.size,
          }

          onDocumentAdd(newDocument)
          simulateFileProcessing(newDocument.id)
        }
      })
    },
    [onDocumentAdd],
  )

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    files.forEach((file) => {
      if (file.type === "application/pdf") {
        const newDocument: UploadedDocument = {
          id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
          name: file.name,
          type: "file",
          source: file.name,
          status: "uploading",
          progress: 0,
          size: file.size,
        }

        onDocumentAdd(newDocument)
        simulateFileProcessing(newDocument.id)
      }
    })
  }

  const simulateFileProcessing = (documentId: string) => {
    let progress = 0
    const uploadInterval = setInterval(() => {
      progress += Math.random() * 20
      if (progress >= 100) {
        progress = 100
        clearInterval(uploadInterval)

        setTimeout(() => {
          console.log(`[v0] Document ${documentId} processing started`)

          let processProgress = 0
          const processInterval = setInterval(() => {
            processProgress += Math.random() * 15
            if (processProgress >= 100) {
              processProgress = 100
              clearInterval(processInterval)

              setTimeout(() => {
                console.log(`[v0] Document ${documentId} processing completed`)
              }, 500)
            }
          }, 200)
        }, 500)
      }
    }, 100)
  }

  const handleUrlSubmit = async () => {
    if (!pdfUrl.trim()) return

    setIsProcessingUrl(true)

    const newDocument: UploadedDocument = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      name: extractFilenameFromUrl(pdfUrl),
      type: "url",
      source: pdfUrl,
      status: "processing",
      progress: 0,
    }

    onDocumentAdd(newDocument)

    setTimeout(() => {
      console.log(`[v0] URL document ${newDocument.id} processing completed`)
      setIsProcessingUrl(false)
      setPdfUrl("")
    }, 2000)
  }

  const extractFilenameFromUrl = (url: string): string => {
    try {
      const pathname = new URL(url).pathname
      const filename = pathname.split("/").pop() || "document.pdf"
      return filename.endsWith(".pdf") ? filename : filename + ".pdf"
    } catch {
      return "document.pdf"
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const getStatusIcon = (status: UploadedDocument["status"]) => {
    switch (status) {
      case "uploading":
      case "processing":
        return <Loader2 className="h-4 w-4 animate-spin" />
      case "ready":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-500" />
    }
  }

  const getStatusText = (status: UploadedDocument["status"]) => {
    switch (status) {
      case "uploading":
        return "Uploading..."
      case "processing":
        return "Processing..."
      case "ready":
        return "Ready"
      case "error":
        return "Error"
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Add Documents</h3>

        <Card>
          <CardContent className="p-6">
            <div className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors border-muted-foreground/25 hover:border-primary/50">
              <input
                type="file"
                accept=".pdf"
                multiple
                onChange={handleFileInput}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <p className="text-lg font-medium mb-2">Upload PDF files</p>
                <p className="text-sm text-muted-foreground">Drag and drop PDF files here, or click to select files</p>
              </label>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <Link className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
                <Input
                  placeholder="Enter PDF URL (e.g., https://arxiv.org/pdf/1234.5678.pdf)"
                  value={pdfUrl}
                  onChange={(e) => setPdfUrl(e.target.value)}
                  className="pl-10"
                  onKeyPress={(e) => e.key === "Enter" && handleUrlSubmit()}
                />
              </div>
              <Button onClick={handleUrlSubmit} disabled={!pdfUrl.trim() || isProcessingUrl}>
                {isProcessingUrl ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  "Add URL"
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {documents.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Uploaded Documents</h3>

          <div className="space-y-3">
            {documents.map((doc) => (
              <Card
                key={doc.id}
                className={`cursor-pointer transition-colors ${
                  selectedDocumentId === doc.id ? "ring-2 ring-primary" : "hover:bg-muted/50"
                }`}
                onClick={() => doc.status === "ready" && onDocumentSelect(doc.id)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <FileText className="h-5 w-5 text-muted-foreground mt-0.5" />

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium truncate">{doc.name}</h4>
                        <Badge variant={doc.type === "file" ? "default" : "secondary"}>
                          {doc.type === "file" ? "File" : "URL"}
                        </Badge>
                        <div className="flex items-center gap-1 text-sm text-muted-foreground">
                          {getStatusIcon(doc.status)}
                          <span>{getStatusText(doc.status)}</span>
                        </div>
                      </div>

                      {doc.size && (
                        <p className="text-sm text-muted-foreground mb-2">
                          {formatFileSize(doc.size)}
                          {doc.pages && ` • ${doc.pages} pages`}
                        </p>
                      )}

                      {(doc.status === "uploading" || doc.status === "processing") && (
                        <Progress value={doc.progress} className="h-2 mb-2" />
                      )}

                      {doc.error && (
                        <Alert className="mt-2">
                          <AlertCircle className="h-4 w-4" />
                          <AlertDescription>{doc.error}</AlertDescription>
                        </Alert>
                      )}

                      {doc.metadata?.title && doc.status === "ready" && (
                        <div className="mt-2 p-2 bg-muted/50 rounded text-sm">
                          <p className="font-medium">{doc.metadata.title}</p>
                          {doc.metadata.authors && (
                            <p className="text-muted-foreground">{doc.metadata.authors.join(", ")}</p>
                          )}
                        </div>
                      )}
                    </div>

                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        onDocumentRemove(doc.id)
                      }}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {documents.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No documents uploaded yet</p>
          <p className="text-sm">Upload PDF files or add URLs to get started</p>
        </div>
      )}
    </div>
  )
}
