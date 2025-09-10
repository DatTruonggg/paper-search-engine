'use client';

import { Plus } from 'lucide-react';

export default function Hero() {
  return (
    <div className="text-center space-y-6 py-12">
      {/* Logo */}
      <div className="flex items-center justify-center space-x-2 text-4xl font-bold">
        <Plus className="h-8 w-8 text-blue-500" />
        <span className="text-blue-400">Literature</span>
      </div>
      
      {/* Description */}
      <div className="max-w-4xl mx-auto space-y-4">
        <p className="text-xl text-muted-foreground">
          A scholarly research assistant with broad and deep coverage
        </p>
        
        <p className="text-lg text-muted-foreground">
          This chatbot can make mistakes 
          A project from <span className="text-blue-400">Dat and Tin</span>.
        </p>
      </div>
    </div>
  );
}
