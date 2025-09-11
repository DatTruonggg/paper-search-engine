# Paper Search Engine

A simple academic paper search engine inspired by ASTA (AllenAI) paper finder, built with FastAPI backend and Streamlit UI for proof-of-concept demonstrations.

## Features

- **Hybrid Search System**: Integrates official ASTA Paper Finder with custom implementation
- **Natural Language Search**: Search for papers using natural language queries
- **Multi-Agent System**: 
  - Query Analyzer Agent: Understands search intent and extracts keywords
  - Hybrid Searcher Agent: Uses official ASTA when available, falls back to custom Semantic Scholar implementation
  - ASTA Paper Finder: Official AllenAI paper search agent with advanced capabilities
- **Multiple Search Methods**:
  - **ASTA Official**: Advanced paper finding with sophisticated relevance scoring
  - **Custom S2**: Fast implementation using Semantic Scholar API
- **Fast & Diligent Modes**: Choose between quick results or thorough searches
- **Smart Filtering**: Filter by year, venue, authors, and open access availability
- **Export Results**: Download search results as CSV
- **Real-time Status**: Shows which search methods are available
- **Clean UI**: Simple Streamlit interface for easy demonstrations

## Architecture

```
┌─────────────────────────┐
│    Streamlit UI         │
│  (User Interface)       │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│    FastAPI Backend      │
│   (API Gateway)         │
└───────────┬─────────────┘
            │
    ┌───────┴────────┐
    ▼                ▼
┌─────────┐    ┌──────────┐
│ Query   │    │ Paper    │
│Analyzer │    │ Searcher │
│ Agent   │    │  Agent   │
└─────────┘    └──────────┘
    │                │
    ▼                ▼
┌─────────┐    ┌──────────┐
│ OpenAI  │    │ Semantic │
│   API   │    │ Scholar  │
└─────────┘    └──────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- **Required API Keys**:
  - OpenAI API key (required for all functionality)
- **Optional API Keys**:
  - Semantic Scholar API key (recommended for higher rate limits)
  - Cohere API key (only needed for full ASTA integration)
  - Google API key (only needed for full ASTA integration)
  - ASTA Tool Key (for enhanced ASTA features)

### Installation

1. Clone the repository:
```bash
git clone <your-repo>
cd paper-search-engine
```

2. **Setup official ASTA packages** (recommended):
```bash
./setup_asta.sh
```
This will clone the official ASTA Paper Finder and Agent Baselines repositories.

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# - OPENAI_API_KEY (required)
# - S2_API_KEY (optional but recommended)
# For full ASTA integration, also uncomment and add:
# - COHERE_API_KEY (optional)
# - GOOGLE_API_KEY (optional)
```

4. Create and activate virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

5. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Locally

#### Option 1: Automated startup with virtual environment (Recommended)

```bash
./run.sh
```

This script will:
- Create a virtual environment (if not exists)
- Activate the virtual environment
- Install dependencies (if needed)
- Start both backend and frontend services
- Handle graceful shutdown on Ctrl+C

#### Option 2: Run services manually

1. Activate virtual environment (if created):
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Start the backend:
```bash
python -m uvicorn backend.api.main:app --reload --port 8000
```

3. In a new terminal, start Streamlit:
```bash
source venv/bin/activate  # Activate venv in new terminal
streamlit run streamlit_app/app.py
```

4. Open your browser to http://localhost:8501

#### Option 2: Full ASTA Integration (Advanced)

For advanced users who want to use the official ASTA Paper Finder:

```bash
./run_with_asta.sh
```

This will start both ASTA server and our application. See [ASTA_SETUP_GUIDE.md](ASTA_SETUP_GUIDE.md) for detailed instructions.

#### Option 3: Docker with ASTA (Recommended for Production)

**🐳 Easy Docker setup with full ASTA integration:**

```bash
# Quick setup with Docker
cp .env.example .env
echo "OPENAI_API_KEY=your_key" >> .env

# Start everything with one command
./docker/manage-asta.sh start-full
```

This provides the complete stack: Backend + Frontend + ASTA + Redis + PostgreSQL

#### Option 4: Docker ASTA Only

**Start just ASTA services to use with local development:**

```bash
./docker/manage-asta.sh start    # Start ASTA services (setup is automatic)
./run.sh                         # Run our app locally
```

#### Option 5: Traditional Docker Compose

```bash
# Basic setup without ASTA
cd docker && docker-compose up --build
```

**Access URLs:**
- **Streamlit UI**: http://localhost:8501
- **Backend API**: http://localhost:8000  
- **API Documentation**: http://localhost:8000/docs
- **ASTA Server** (if using Docker): http://localhost:8080
- **Search Capabilities**: http://localhost:8000/api/capabilities

## Configuration

Edit `.env` file to configure:

```env
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional but recommended
S2_API_KEY=your_semantic_scholar_api_key

# Search settings
MAX_SEARCH_RESULTS=20
DEFAULT_SEARCH_MODE=fast
SEARCH_TIMEOUT_SECONDS=30

# LLM settings
LLM_MODEL=gpt-4o-mini
TEMPERATURE=0.7
```

## API Endpoints

### Health Check
```
GET /health
```

### Search Papers
```
POST /api/search
{
  "query": "transformer architecture for NLP",
  "mode": "fast",
  "max_results": 20,
  "filters": {
    "year_range": {"start": 2020, "end": 2024},
    "venues": ["ACL", "EMNLP"],
    "open_access_only": true
  }
}
```

## Example Queries

- "Recent papers on large language models"
- "Transformer architecture surveys from 2020 to 2024"
- "Papers by Yoshua Bengio on deep learning"
- "Influential papers on reinforcement learning"
- "BERT and its applications in NLP"

## Development

### Project Structure
```
paper-search-engine/
├── backend/
│   ├── agents/           # Agent implementations
│   ├── api/             # FastAPI application
│   ├── services/        # External service integrations
│   └── config.py        # Configuration
├── streamlit_app/       # Streamlit UI
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Docker orchestration
└── .env.example        # Environment variables template
```

### Testing

Run tests (when implemented):
```bash
pytest backend/tests/
```

## Roadmap

- [ ] Add caching with Redis
- [ ] Implement more sophisticated ranking algorithms
- [ ] Add citation network exploration
- [ ] Support for full-text search
- [ ] Advanced filters (citation count, publication type)
- [ ] User authentication and search history
- [ ] Integration with more academic databases

## Credits

Inspired by:
- [ASTA Paper Finder](https://github.com/allenai/asta-paper-finder) by AllenAI
- [Agent Baselines](https://github.com/allenai/agent-baselines) by AllenAI

## License

MIT
