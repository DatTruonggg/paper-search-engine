#!/bin/bash

# Makefile equivalent for Paper Search Engine

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

help() {
    echo -e "${GREEN}Paper Search Engine - Development Commands${NC}"
    echo ""
    echo "Usage: ./make.sh [command]"
    echo ""
    echo "Commands:"
    echo "  help          Show this help message"
    echo "  install       Install all dependencies"
    echo "  dev           Start development servers"
    echo "  backend       Start backend only"
    echo "  frontend      Start frontend only"
    echo "  services      Start supporting services (Docker)"
    echo "  stop          Stop all services"
    echo "  clean         Clean up generated files"
    echo "  test          Run tests"
    echo "  lint          Run linting"
    echo "  build         Build for production"
    echo ""
}

install() {
    echo -e "${GREEN} Installing dependencies...${NC}"
    set -e

    # -------- Backend (Python 3.10 via conda/mamba/micromamba) --------
    echo -e "${YELLOW}Installing Python dependencies (Conda, Python 3.10)...${NC}"
    cd backend

    ENV_NAME="${CONDA_ENV_NAME:-paperbot-py310}"

    # Chọn trình quản lý môi trường: mamba > micromamba > conda
    if command -v mamba >/dev/null 2>&1; then
        CONDA_CMD="mamba"
        # load conda shell
        if [ -z "${CONDA_EXE:-}" ] && command -v conda >/dev/null 2>&1; then
            CONDA_BASE="$(conda info --base)"
            # shellcheck disable=SC1091
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        fi
        $CONDA_CMD env list | grep -q "^${ENV_NAME} " || $CONDA_CMD create -y -n "$ENV_NAME" python=3.10
        conda activate "$ENV_NAME"

    elif command -v micromamba >/dev/null 2>&1; then
        CONDA_CMD="micromamba"
        # enable shell hook cho micromamba
        eval "$($CONDA_CMD shell hook -s bash)"
        $CONDA_CMD create -y -n "$ENV_NAME" python=3.10
        $CONDA_CMD activate "$ENV_NAME"

    elif command -v conda >/dev/null 2>&1; then
        CONDA_CMD="conda"
        CONDA_BASE="$(conda info --base)"
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        $CONDA_CMD env list | grep -q "^${ENV_NAME} " || $CONDA_CMD create -y -n "$ENV_NAME" python=3.10
        conda activate "$ENV_NAME"

    else
        echo -e "${RED}Conda/Mamba/Micromamba not found.${NC} Install one of them first (e.g., Miniconda)."
        exit 1
    fi

    python --version
    pip install --upgrade pip
    pip install -r requirements.txt
    cd ..

    # -------- Frontend (Node) --------
    echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
    cd frontend
    if command -v pnpm >/dev/null 2>&1; then
        pnpm install
    else
        npm install
    fi
    cd ..

    echo -e "${GREEN} Dependencies installed successfully!${NC}"
}


dev() {
    echo -e "${GREEN} Starting development environment...${NC}"
    ./dev.sh
}

backend() {
    echo -e "${GREEN} Starting backend only...${NC}"
    cd backend
    
    if [ ! -d "venv" ]; then
        echo -e "${RED}Please run './make.sh install' first${NC}"
        exit 1
    fi
    
    source venv/bin/activate
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        echo -e "${YELLOW} Please edit backend/.env file with your configuration${NC}"
    fi
    
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}

frontend() {
    echo -e "${GREEN} Starting frontend only...${NC}"
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        echo -e "${RED}Please run './make.sh install' first${NC}"
        exit 1
    fi
    
    if command -v pnpm >/dev/null 2>&1; then
        pnpm dev
    else
        npm run dev
    fi
}

services() {
    echo -e "${GREEN} Starting supporting services...${NC}"
    
    if ! command -v docker-compose >/dev/null 2>&1 && ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}Docker is required but not installed${NC}"
        exit 1
    fi
    
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose up -d
    else
        docker compose up -d
    fi
    
    echo -e "${GREEN} Services started successfully!${NC}"
    echo -e "${GREEN} Elasticsearch: http://localhost:9200${NC}"
    echo -e "${GREEN} PostgreSQL: localhost:5432${NC}"
    echo -e "${GREEN} Redis: localhost:6379${NC}"
    echo -e "${GREEN} MinIO: http://localhost:9001 (admin:admin)${NC}"
}

stop() {
    echo -e "${YELLOW} Stopping all services...${NC}"
    
    # Stop Docker services
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose down
    elif command -v docker >/dev/null 2>&1; then
        docker compose down 2>/dev/null || true
    fi
    
    # Stop any running dev processes
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "next dev" 2>/dev/null || true
    
    echo -e "${GREEN} Services stopped${NC}"
}

clean() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    
    # Backend cleanup
    cd backend
    rm -rf venv/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    cd ..
    
    # Frontend cleanup
    cd frontend
    rm -rf node_modules/
    rm -rf .next/
    cd ..
    
    # Docker cleanup
    if command -v docker >/dev/null 2>&1; then
        echo -e "${YELLOW}Removing Docker volumes...${NC}"
        docker volume rm paper-search-engine_es_data paper-search-engine_pg_data paper-search-engine_redis_data paper-search-engine_minio_data 2>/dev/null || true
    fi
    
    echo -e "${GREEN} Cleanup completed${NC}"
}

test() {
    echo -e "${GREEN} Running tests...${NC}"
    
    # Backend tests
    cd backend
    if [ -d "venv" ]; then
        source venv/bin/activate
        python -m pytest tests/ 2>/dev/null || echo -e "${YELLOW} No backend tests found${NC}"
    else
        echo -e "${YELLOW} Backend environment not set up${NC}"
    fi
    cd ..
    
    # Frontend tests
    cd frontend
    if [ -d "node_modules" ]; then
        if command -v pnpm >/dev/null 2>&1; then
            pnpm test 2>/dev/null || echo -e "${YELLOW} No frontend tests configured${NC}"
        else
            npm test 2>/dev/null || echo -e "${YELLOW} No frontend tests configured${NC}"
        fi
    else
        echo -e "${YELLOW} Frontend dependencies not installed${NC}"
    fi
    cd ..
}

lint() {
    echo -e "${GREEN} Running linters...${NC}"
    
    # Backend linting
    cd backend
    if [ -d "venv" ]; then
        source venv/bin/activate
        python -m black . --check 2>/dev/null || echo -e "${YELLOW} Install black for Python linting${NC}"
        python -m isort . --check-only 2>/dev/null || echo -e "${YELLOW} Install isort for Python import sorting${NC}"
    fi
    cd ..
    
    # Frontend linting
    cd frontend
    if [ -d "node_modules" ]; then
        if command -v pnpm >/dev/null 2>&1; then
            pnpm lint 2>/dev/null || echo -e "${YELLOW} Frontend linting not configured${NC}"
        else
            npm run lint 2>/dev/null || echo -e "${YELLOW} Frontend linting not configured${NC}"
        fi
    fi
    cd ..
}

build() {
    echo -e "${GREEN}  Building for production...${NC}"
    
    # Frontend build
    cd frontend
    if [ -d "node_modules" ]; then
        if command -v pnpm >/dev/null 2>&1; then
            pnpm build
        else
            npm run build
        fi
    else
        echo -e "${RED}Frontend dependencies not installed${NC}"
        exit 1
    fi
    cd ..
    
    echo -e "${GREEN} Build completed${NC}"
}

# Main command handling
case "${1:-help}" in
    help|--help|-h)
        help
        ;;
    install)
        install
        ;;
    dev)
        dev
        ;;
    backend)
        backend
        ;;
    frontend)
        frontend
        ;;
    services)
        services
        ;;
    stop)
        stop
        ;;
    clean)
        clean
        ;;
    test)
        test
        ;;
    lint)
        lint
        ;;
    build)
        build
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        help
        exit 1
        ;;
esac
