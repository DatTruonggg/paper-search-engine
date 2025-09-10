import re
from typing import List, Dict, Any


class TokenizerService:
    def __init__(self):
        # Common stopwords
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how',
            'their', 'if', 'up', 'out', 'many', 'then', 'them'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into meaningful terms"""
        if not text:
            return []
        
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # Filter out stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 2]
        
        return tokens
    
    def get_why_shown(self, query: str, paper_data: Dict[str, Any]) -> List[str]:
        """Determine why a paper was shown in search results"""
        why_shown = []
        
        if not query:
            return why_shown
        
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return why_shown
        
        # Check title matches
        title_tokens = self.tokenize(paper_data.get("title", ""))
        title_matches = [token for token in query_tokens if token in title_tokens]
        if title_matches:
            why_shown.append(f"Title matches: {', '.join(title_matches)}")
        
        # Check abstract matches
        abstract_tokens = self.tokenize(paper_data.get("abstract", ""))
        abstract_matches = [token for token in query_tokens if token in abstract_tokens]
        if abstract_matches:
            why_shown.append(f"Abstract matches: {', '.join(abstract_matches)}")
        
        # Check author matches
        authors_text = " ".join(paper_data.get("authors", []))
        author_tokens = self.tokenize(authors_text)
        author_matches = [token for token in query_tokens if token in author_tokens]
        if author_matches:
            why_shown.append(f"Author matches: {', '.join(author_matches)}")
        
        # Check category matches
        categories_text = " ".join(paper_data.get("categories", []))
        category_tokens = self.tokenize(categories_text)
        category_matches = [token for token in query_tokens if token in category_tokens]
        if category_matches:
            why_shown.append(f"Category matches: {', '.join(category_matches)}")
        
        # If no specific matches found but paper was returned, it's likely a fuzzy match
        if not why_shown:
            why_shown.append("Fuzzy text match")
        
        return why_shown
