'use client';

import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import Hero from '@/components/Hero';
import SearchBar from '@/components/SearchBar';
import PaperCard from '@/components/PaperCard';
import { api, ApiError } from '@/lib/api';
import { SearchRequest, SearchResponse, ChatResponse } from '@/lib/types';

export default function HomePage() {
  const [isSearching, setIsSearching] = useState(false);
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isIngestingData, setIsIngestingData] = useState(false);

  const handleSearch = async (query: string) => {
    setIsSearching(true);
    setError(null);
    setChatResponse(null);

    try {
      const request: SearchRequest = {
        q: query,
        page: 1,
        pageSize: 20,
        sort: 'relevance'
      };

      const response = await api.search(request);
      setSearchResponse(response);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred during search');
      }
      console.error('Search error:', err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSummarize = async (paperId: string) => {
    if (!searchResponse) return;

    try {
      const paper = searchResponse.results.find(p => p.id === paperId);
      if (!paper) return;

      const chatRequest = {
        message: `Please provide a comprehensive summary of this paper: "${paper.title}" by ${paper.authors.slice(0, 3).join(', ')}. Include the main research question, methodology, key findings, and significance.`,
        topK: 1,
        history: []
      };

      const response = await api.chat(chatRequest);
      setChatResponse(response);
    } catch (err) {
      console.error('Summarize error:', err);
      setError('Failed to generate summary');
    }
  };

  const handleIngestData = async () => {
    setIsIngestingData(true);
    setError(null);

    try {
      const response = await api.ingest(false, 1000);
      alert(`Data ingestion completed: ${JSON.stringify(response)}`);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Data ingestion failed');
      }
      console.error('Ingest error:', err);
    } finally {
      setIsIngestingData(false);
    }
  };

  const handleLoadMore = async () => {
    if (!searchResponse) return;

    setIsSearching(true);
    try {
      const request: SearchRequest = {
        q: searchResponse.analysis.query,
        page: searchResponse.page + 1,
        pageSize: 20,
        sort: searchResponse.analysis.sort as 'relevance' | 'recency'
      };

      const response = await api.search(request);
      setSearchResponse({
        ...response,
        results: [...searchResponse.results, ...response.results]
      });
    } catch (err) {
      console.error('Load more error:', err);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-bold text-blue-400">Paper Search Engine</h1>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleIngestData}
              disabled={isIngestingData}
              className="text-xs"
            >
              {isIngestingData ? 'Ingesting...' : 'Ingest Data'}
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        {!searchResponse && !chatResponse && (
          <Hero />
        )}

        {/* Search Interface */}
        <div className="max-w-6xl mx-auto space-y-8">
          <SearchBar onSearch={handleSearch} isLoading={isSearching} />

          {/* Tabs */}
          <Tabs defaultValue="papers" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="papers">Find papers</TabsTrigger>
              <TabsTrigger value="summarize" disabled>Summarize literature</TabsTrigger>
              <TabsTrigger value="analyze" disabled>Analyze data</TabsTrigger>
            </TabsList>

            <TabsContent value="papers" className="mt-8">
              {/* Error Display */}
              {error && (
                <div className="bg-destructive/10 border border-destructive/20 text-destructive px-4 py-3 rounded-md mb-6">
                  {error}
                </div>
              )}

              {/* Chat Response */}
              {chatResponse && (
                <div className="bg-card border border-border rounded-lg p-6 mb-6">
                  <h3 className="text-lg font-semibold mb-4">Summary</h3>
                  <div className="prose prose-invert max-w-none">
                    <p className="text-foreground leading-relaxed whitespace-pre-wrap">
                      {chatResponse.answer}
                    </p>
                  </div>
                  
                  {chatResponse.citations.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-border">
                      <h4 className="font-medium mb-2">Citations:</h4>
                      <div className="flex flex-wrap gap-2">
                        {chatResponse.citations.map((citation, idx) => (
                          <span 
                            key={idx}
                            className="text-xs bg-blue-900/30 text-blue-300 px-2 py-1 rounded"
                          >
                            {citation.id} {citation.doi && `(${citation.doi})`}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Search Results */}
              {searchResponse && (
                <div className="space-y-6">
                  {/* Results Header */}
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-muted-foreground">
                      {searchResponse.total.toLocaleString()} papers found in {searchResponse.tookMs}ms
                    </div>
                  </div>

                  {/* Results List */}
                  <div className="grid gap-4">
                    {searchResponse.results.map((paper) => (
                      <PaperCard 
                        key={paper.id} 
                        paper={paper}
                        onSummarize={handleSummarize}
                      />
                    ))}
                  </div>

                  {/* Load More */}
                  {searchResponse.results.length < searchResponse.total && (
                    <div className="flex justify-center pt-6">
                      <Button 
                        onClick={handleLoadMore}
                        disabled={isSearching}
                        variant="outline"
                      >
                        {isSearching ? 'Loading...' : 'Load More Results'}
                      </Button>
                    </div>
                  )}
                </div>
              )}

              {/* Empty State */}
              {!searchResponse && !isSearching && !error && (
                <div className="text-center py-12">
                  <p className="text-muted-foreground">
                    Enter a search query above to find academic papers
                  </p>
                  <div className="mt-6 space-y-2">
                    <p className="text-sm text-muted-foreground">Example searches:</p>
                    <div className="flex flex-wrap justify-center gap-2">
                      {['diphoton', 'machine learning', 'quantum computing', 'climate change'].map(example => (
                        <Button
                          key={example}
                          variant="outline"
                          size="sm"
                          onClick={() => handleSearch(example)}
                          className="text-xs"
                        >
                          {example}
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-sm text-muted-foreground">
            <p>Paper Search Engine - Built with FastAPI and Next.js</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
