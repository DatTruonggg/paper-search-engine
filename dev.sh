#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN} Starting Paper Search Engine...${NC}"

# Check if we're in the project root
if [ ! -f "README.md" ]; then
    echo -e "${RED} Please run this script from the project root directory${NC}"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo -e "${YELLOW} Checking dependencies...${NC}"

if ! command_exists python3; then
    echo -e "${RED} Python 3 is required but not installed${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED} Node.js is required but not installed${NC}"
    exit 1
fi

if ! command_exists pnpm; then
    echo -e "${YELLOW}  pnpm not found, using npm instead${NC}"
    NPM_CMD="npm"
else
    NPM_CMD="pnpm"
fi

# Start backend
echo -e "${GREEN} Starting backend...${NC}"
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}  Please edit backend/.env file with your configuration${NC}"
fi

# Start backend in background
echo -e "${GREEN}Starting FastAPI server...${NC}"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

cd ..

# Start frontend
echo -e "${GREEN} Starting frontend...${NC}"
cd frontend

# Install Node dependencies
echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
$NPM_CMD install

# Start frontend in background
echo -e "${GREEN}Starting Next.js server...${NC}"
$NPM_CMD dev &
FRONTEND_PID=$!

cd ..

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW} Cleaning up...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

echo -e "${GREEN} Services started successfully!${NC}"
echo -e "${GREEN} Frontend: http://localhost:3000${NC}"
echo -e "${GREEN} Backend API: http://localhost:8000${NC}"
echo -e "${GREEN} API Docs: http://localhost:8000/docs${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Wait for user interrupt
wait
