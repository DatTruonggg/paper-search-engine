import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Any, Optional
import time


# Page configuration
st.set_page_config(
    page_title="Paper Search Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
API_URL = "http://localhost:8000"


def check_backend_health():
    """Check if backend is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None


def get_search_capabilities():
    """Get available search capabilities."""
    try:
        response = requests.get(f"{API_URL}/api/capabilities", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def search_papers(query: str, mode: str = "fast", max_results: int = 20, filters: Optional[Dict] = None) -> Dict:
    """Search for papers using the backend API."""
    try:
        payload = {
            "query": query,
            "mode": mode,
            "max_results": max_results
        }
        
        if filters:
            payload["filters"] = filters
        
        response = requests.post(
            f"{API_URL}/api/search",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "error": f"API returned status code {response.status_code}",
                "papers": [],
                "count": 0
            }
    
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error": "Search request timed out. Please try again.",
            "papers": [],
            "count": 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "papers": [],
            "count": 0
        }


def display_paper_card(paper: Dict):
    """Display a paper as a card."""
    with st.container():
        # Title
        st.markdown(f"### {paper.get('title', 'Unknown Title')}")
        
        # Authors
        authors = paper.get('authors', [])
        if authors:
            author_names = [author.get('name', 'Unknown') for author in authors[:5]]
            if len(authors) > 5:
                author_names.append(f"... and {len(authors) - 5} more")
            st.markdown(f"**Authors:** {', '.join(author_names)}")
        
        # Metadata row
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            year = paper.get('year', 'Unknown')
            st.markdown(f"**Year:** {year}")
        with col2:
            citations = paper.get('citationCount', 0) or 0
            st.markdown(f"**Citations:** {citations:,}")
        with col3:
            venue = paper.get('venue', 'Unknown venue')
            if venue:
                st.markdown(f"**Venue:** {venue}")
        
        # Abstract
        abstract = paper.get('abstract', None)
        tldr = paper.get('tldr', {})
        
        if tldr and tldr.get('text'):
            with st.expander("TL;DR"):
                st.write(tldr['text'])
        
        if abstract:
            with st.expander("Abstract"):
                st.write(abstract)
        
        # Links
        col1, col2 = st.columns(2)
        with col1:
            paper_id = paper.get('paperId', '')
            if paper_id:
                st.markdown(f"[View on Semantic Scholar](https://www.semanticscholar.org/paper/{paper_id})")
        
        with col2:
            pdf_info = paper.get('openAccessPdf', {})
            if pdf_info and pdf_info.get('url'):
                st.markdown(f"[📄 PDF]({pdf_info['url']})")
        
        st.divider()


def export_to_csv(papers: List[Dict]) -> pd.DataFrame:
    """Convert papers to DataFrame for export."""
    data = []
    for paper in papers:
        authors = paper.get('authors', [])
        author_names = ', '.join([a.get('name', '') for a in authors])
        
        data.append({
            'Title': paper.get('title', ''),
            'Authors': author_names,
            'Year': paper.get('year', ''),
            'Venue': paper.get('venue', ''),
            'Citations': paper.get('citationCount', 0),
            'Abstract': paper.get('abstract', ''),
            'Paper ID': paper.get('paperId', ''),
            'PDF URL': paper.get('openAccessPdf', {}).get('url', '') if paper.get('openAccessPdf') else ''
        })
    
    return pd.DataFrame(data)


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("📚 Paper Search Engine")
    st.markdown("Search for academic papers using natural language queries")
    
    # Check backend health and capabilities
    backend_healthy, health_info = check_backend_health()
    capabilities = get_search_capabilities() if backend_healthy else None
    
    # Sidebar
    with st.sidebar:
        st.header("Search Settings")
        
        # Backend status
        if backend_healthy:
            st.success("✅ Backend Connected")
            
            # Show search methods available
            if capabilities:
                st.subheader("🔍 Search Methods")
                preferred = capabilities.get("preferred_method", "unknown")
                
                # Check for mock mode
                if capabilities.get("mock_available"):
                    st.success("🎭 Mock Database: Available")
                    if preferred == "mock_database":
                        st.caption("⭐ Primary method (Demo Mode)")
                
                if capabilities.get("asta_available"):
                    st.success("🎯 ASTA Official: Available")
                    if preferred == "asta_official":
                        st.caption("⭐ Primary method")
                else:
                    st.warning("🎯 ASTA Official: Unavailable")
                
                st.info("🔧 Custom S2: Available")
                if preferred == "custom_s2":
                    st.caption("⭐ Primary method")
                
                # Show configuration info if available
                config = capabilities.get("configuration", {})
                if config.get("mock_mode"):
                    st.info("🎭 Running in Demo Mode - using mock data")
                    
                # Show service status
                if health_info and "services" in health_info:
                    with st.expander("Service Status"):
                        services = health_info["services"]
                        for service, status in services.items():
                            if status == "connected" or status == "available":
                                st.success(f"✅ {service}: {status}")
                            elif status == "unavailable":
                                st.warning(f"⚠️ {service}: {status}")
                            else:
                                st.error(f"❌ {service}: {status}")
            
        else:
            st.error("❌ Backend Disconnected")
            st.info("Please ensure the backend is running on http://localhost:8000")
        
        # Search mode
        search_mode = st.selectbox(
            "Search Mode",
            options=["fast", "diligent"],
            help="Fast mode returns results quickly. Diligent mode is more thorough but slower."
        )
        
        # Max results
        max_results = st.slider(
            "Maximum Results",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        
        st.divider()
        
        # Filters
        st.subheader("Filters")
        
        # Year filter
        use_year_filter = st.checkbox("Filter by year")
        year_start = None
        year_end = None
        if use_year_filter:
            col1, col2 = st.columns(2)
            with col1:
                year_start = st.number_input("From", min_value=1900, max_value=2024, value=2020)
            with col2:
                year_end = st.number_input("To", min_value=1900, max_value=2024, value=2024)
        
        # Open access filter
        open_access_only = st.checkbox("Open Access Only")
        
        # Venue filter
        use_venue_filter = st.checkbox("Filter by venue")
        venues = []
        if use_venue_filter:
            venue_input = st.text_area(
                "Venues (one per line)",
                placeholder="ACL\nEMNLP\nNeurIPS",
                help="Enter conference/journal names, one per line"
            )
            if venue_input:
                venues = [v.strip() for v in venue_input.split('\n') if v.strip()]
    
    # Main search interface
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="e.g., 'transformer architecture for natural language processing'",
            help="Enter a natural language query to search for papers"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Example queries
    with st.expander("Example Queries"):
        st.markdown("""
        - Recent papers on large language models
        - Transformer architecture surveys from 2020 to 2024
        - Papers by Yoshua Bengio on deep learning
        - Influential papers on reinforcement learning
        - BERT and its applications in NLP
        """)
    
    # Search execution
    if search_button and query:
        if not backend_healthy:
            st.error("Cannot search: Backend is not connected. Please start the backend server.")
        else:
            # Prepare filters
            filters = {}
            if use_year_filter and year_start and year_end:
                filters["year_range"] = {"start": year_start, "end": year_end}
            if open_access_only:
                filters["open_access_only"] = True
            if venues:
                filters["venues"] = venues
            
            # Show search progress
            with st.spinner(f"Searching for papers... (Mode: {search_mode})"):
                start_time = time.time()
                results = search_papers(query, search_mode, max_results, filters if filters else None)
                search_time = time.time() - start_time
            
            # Display results
            if results["status"] == "success":
                st.success(f"Found {results['count']} papers in {search_time:.2f} seconds")
                
                # Display search metadata
                if results.get("search_metadata"):
                    metadata = results["search_metadata"]
                    with st.expander("Search Details"):
                        st.json(metadata)
                
                # Export button
                if results["papers"]:
                    df = export_to_csv(results["papers"])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"paper_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Display papers
                st.divider()
                for i, paper in enumerate(results["papers"], 1):
                    st.markdown(f"**Result {i}**")
                    display_paper_card(paper)
            
            elif results["status"] == "error":
                st.error(f"Search failed: {results.get('error', 'Unknown error')}")
                if results.get('message'):
                    st.info(results['message'])
            
            else:
                st.warning("No papers found matching your query. Try different keywords or broaden your search.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Paper Search Engine - Powered by ASTA-inspired agents and Semantic Scholar API
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()