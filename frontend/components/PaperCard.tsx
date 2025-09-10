'use client';

import { PaperResult } from '@/lib/types';
import { ExternalLink, Users, Calendar, Tag } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface PaperCardProps {
  paper: PaperResult;
  onSummarize?: (paperId: string) => void;
}

export default function PaperCard({ paper, onSummarize }: PaperCardProps) {
  const handlePdfClick = () => {
    if (paper.urlPdf) {
      window.open(paper.urlPdf, '_blank');
    }
  };

  const handleDoiClick = () => {
    if (paper.doi) {
      window.open(`https://doi.org/${paper.doi}`, '_blank');
    }
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6 hover:bg-accent/50 transition-colors">
      <div className="space-y-4">
        {/* Header */}
        <div className="space-y-2">
          <h3 className="text-lg font-semibold text-foreground leading-tight">
            {paper.title}
          </h3>
          
          <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Users className="h-4 w-4" />
              <span>{paper.authors.slice(0, 3).join(', ')}</span>
              {paper.authors.length > 3 && <span> +{paper.authors.length - 3} more</span>}
            </div>
            
            <div className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              <span>{paper.year}</span>
            </div>
            
            <div className="flex items-center gap-1">
              <Tag className="h-4 w-4" />
              <span className="text-xs bg-secondary px-2 py-1 rounded">
                Score: {paper.score.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Abstract */}
        <div className="space-y-2">
          <p className="text-sm text-foreground leading-relaxed">
            {paper.abstractSnippet}
          </p>
          
          {paper.whyShown.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {paper.whyShown.map((reason, idx) => (
                <span 
                  key={idx}
                  className="text-xs bg-blue-900/50 text-blue-300 px-2 py-1 rounded-full"
                >
                  {reason}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Categories */}
        <div className="flex flex-wrap gap-2">
          {paper.categories.slice(0, 4).map((category) => (
            <span 
              key={category}
              className="text-xs bg-muted text-muted-foreground px-2 py-1 rounded"
            >
              {category}
            </span>
          ))}
          {paper.categories.length > 4 && (
            <span className="text-xs text-muted-foreground">
              +{paper.categories.length - 4} more
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-wrap items-center gap-2 pt-2">
          {paper.urlPdf && (
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handlePdfClick}
              className="text-xs"
            >
              <ExternalLink className="h-3 w-3 mr-1" />
              PDF
            </Button>
          )}
          
          {paper.doi && (
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleDoiClick}
              className="text-xs"
            >
              <ExternalLink className="h-3 w-3 mr-1" />
              DOI
            </Button>
          )}
          
          {onSummarize && (
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => onSummarize(paper.id)}
              className="text-xs bg-blue-900/20 hover:bg-blue-800/30 border-blue-600/30"
            >
              Summarize
            </Button>
          )}
          
          <div className="flex-1" />
          
          <span className="text-xs text-muted-foreground">
            ArXiv: {paper.id}
          </span>
        </div>
      </div>
    </div>
  );
}
