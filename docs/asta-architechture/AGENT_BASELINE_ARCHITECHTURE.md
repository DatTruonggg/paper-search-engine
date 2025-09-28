# Agent-Baselines Codebase Analysis

## 🏗️ **1. Big Picture Architecture & Purpose**

### **Project Overview**
This is a **baseline agent evaluation repository** for [AstaBench](https://github.com/allenai/asta-bench), containing 13 different AI agent implementations. The project serves as:
- **Benchmark baseline collection** for comparing AI agent performance
- **Research platform** for scientific AI agents (literature review, data analysis, paper finding)
- **Evaluation framework** using reproducible pipelines

### **Core Technology Stack**
```
┌─ InspectAI (Agent Framework)
├─ AstaBench (Evaluation & Tools)  
├─ DVC (Pipeline Management)
└─ Docker (Environment Isolation)
```

### **Architectural Principles**
- **Solver Pattern**: All agents implement InspectAI's `Solver` interface
- **Tool Integration**: Standardized toolset through AstaBench `ToolsetConfig`
- **Evaluation First**: Built for reproducible, comparable evaluations
- **Modular Design**: Each agent is self-contained with its own dependencies

---

## 📦 **2. Core Package Structure (`agent_baselines/`)**

### **Main Package Layout**
```
agent_baselines/
├── solvers/
│   ├── __init__.py           # Common imports & utilities
│   ├── llm.py               # Base LLM solver
│   ├── futurehouse.py       # FutureHouse integration
│   ├── youcom.py           # You.com search integration
│   ├── lit_tables.py       # Literature table utilities
│   └── [solver_dirs]/      # Individual solver implementations
└── [solver_specific_dirs]   # Solver-specific modules
```

### **Key Base Components**

#### **`llm.py` - Base LLM Solver**
```python
@solver
def llm_with_prompt(system_prompt: str | None = None) -> Solver:
    """Simple solver that just runs LLM with system prompt"""
    chainlist = [generate()]
    if system_prompt:
        system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
        chainlist.insert(0, system_message(system_prompt))
    return chain(chainlist)
```
- **Purpose**: Foundational solver for simple LLM interactions
- **Usage**: Base class for prompt-only agents
- **Key Feature**: Automatic prompt escaping for InspectAI

#### **`__init__.py` - Common Utilities**
```python
from astabench.util.model import normalize_model_name, record_model_usage_with_inspect
from .futurehouse import futurehouse_solver
from .llm import llm_with_prompt
```
- **Model utilities**: Name normalization and usage tracking
- **Common solvers**: Exports for reuse across agents

---

## 🤖 **3. Individual Solver Analysis**

### **General Purpose Agents**

#### **A. ReAct Agent (`solvers/react/`)**
**Location**: `agent_baselines/solvers/react/basic_agent.py`

**Architecture**:
```python
@solver
def instantiated_basic_agent(
    max_steps: int = 10,
    tool_call_format: Literal["text", "json"] = "json",
    with_search_tools: int = 1,
    # ... other tool options
) -> Solver:
```

**Key Features**:
- **Dual Tool Format**: Supports both JSON tool-calling and text-based "CALL FUNCTION" format
- **Configurable Tools**: All AstaBench tools can be enabled/disabled
- **Step Limiting**: Prevents infinite loops with `max_steps`
- **Custom Prompting**: Modified from InspectAI's basic_agent with task-specific prompts

**Text Tool Call Format Example**:
```
CALL FUNCTION:
```json
{
    "function": "search",
    "arguments": {"query": "quantum computing"}
}
```

**Implementation Highlights**:
```python
def extract_text_tool_calls(message: ChatMessageAssistant) -> ChatMessageAssistant:
    """Extract tool calls from message text using regex parsing"""
    for match in re.finditer(r"CALL FUNCTION:(\s*)```json\n(.*?)```", message.text, re.DOTALL):
        # Parse and convert to ToolCall objects
```

#### **B. Smolagents (`solvers/smolagents/`)**
**Location**: `agent_baselines/solvers/smolagents/agent.py`

**Architecture**: Code-generation approach vs tool-calling
```python
from smolagents import CodeAgent
from agent_baselines.solvers.smolagents.llm_wrapper import AsyncOpenAIModel
from agent_baselines.solvers.smolagents.sandbox_wrapper import InspectAiSandboxExecutor
```

**Key Components**:
1. **LLM Wrapper** (`llm_wrapper.py`): Adapts InspectAI models to smolagents interface
2. **Sandbox Wrapper** (`sandbox_wrapper.py`): Executes Python code in controlled environment
3. **Agent** (`agent.py`): Orchestrates code generation and execution

**Workflow**:
```
User Query → Code Generation → Sandbox Execution → Tool Access → Result
```

### **Specialized Asta Science Agents**

#### **C. Scholar QA (`solvers/sqa/`)**
**Location**: `agent_baselines/solvers/sqa/sqa.py`

**Purpose**: Long-form academic question answering with literature retrieval

**Architecture**:
```python
from scholarqa import ScholarQA
from scholarqa.rag.reranker.modal_engine import ModalReranker
from scholarqa.rag.retrieval import PaperFinderWithReranker
```

**Key Components**:
1. **Retrieval System**: PaperFinderWithReranker for academic papers
2. **Reranking**: ModalReranker for relevance scoring
3. **Response Generation**: Table formatting and citation management

**Model Mapping**:
```python
completion_model_map = {
    "claude-3.7": claude_3_7,
    "claude-3.5": claude_3_5, 
    "claude-4.0": claude_4_0,
    "gemini-2.5-pro": gemini_2_5_pro,
}
```

**Response Processing**:
```python
def format_tables(resp: Dict[str, Any]):
    """Convert tables to markdown format with citations"""
    # Processes JSON table format into sections with citations
```

#### **D. DataVoyager (`solvers/datavoyager/`)**
**Location**: `agent_baselines/solvers/datavoyager/`

**Purpose**: Multi-agent data analysis and exploration

**Architecture**: AutoGen-based multi-agent system
```yaml
# Configuration example
agents:
  planner:
    name: "planner"
    description: "Decomposes high-level plans into actionable steps"
    model_client:
      model: "gpt-4o"
      
  programmer:
    name: "programmer"
    description: "Handles coding/implementation tasks"
    system_message: |
      Generate Python code for assigned tasks...
```

**Agent Types**:
- **Planner**: Task decomposition and strategy
- **Programmer**: Code generation and implementation
- **Data Analyst**: Statistical analysis and visualization
- **Coordinator**: Multi-agent orchestration

#### **E. Other Specialized Agents**

**Asta-v0** (`solvers/asta-v0/`): Task router that delegates to specialized agents

**Paper Finder** (`solvers/paper_finder/`): Academic paper search and recommendation  

**Asta Tables** (`solvers/arxivdigestables/`): Literature review table generation

**Asta Code** (`solvers/super/`): Repository reproduction tasks

**E2E Discovery** (`solvers/e2e_discovery/`): End-to-end discovery workflows

### **Research Baselines**

#### **F. STORM (`solvers/storm/`)**
**Purpose**: Knowledge curation for comprehensive report generation
**Integration**: External knowledge-storm package from GitHub

#### **G. FutureHouse (`solvers/futurehouse/`)**  
**Purpose**: Literature review and scientific writing
**Integration**: futurehouse-client package

---

## 🔧 **4. Infrastructure & Tooling**

### **Docker System**
**Configuration**: `Makefile` + `docker/Dockerfile`

**Key Features**:
```makefile
# Solver-specific containers
make shell SOLVER=react     # React agent environment
make shell SOLVER=sqa       # SQA agent with dependencies

# Development targets  
make test                   # Basic tests
make test-expensive         # API-requiring tests
make format                 # Black formatting
make mypy                   # Type checking
```

**Environment Management**:
```makefile
# Automatic environment variable passing
ENV_ARGS += -e OPENAI_API_KEY
ENV_ARGS += -e ANTHROPIC_API_KEY  
ENV_ARGS += -e ASTA_TOOL_KEY
```

### **DVC Evaluation Pipeline**
**Configuration**: `dvc.yaml`

**Pipeline Structure**:
```yaml
stages:
  solve_sqa:
    matrix:
      model: [claude-3.7, gemini-2.5-pro, claude-4.0, o3_high]
      split: [dev, test]
    cmd: |
      uv run --extra sqa inspect eval astabench/evals/sqa/task.py@sqa
      --solver agent_baselines/solvers/sqa/sqa.py@sqa_solver
      -S completion_model=${item.model}
      --limit=${limit}
```

**Key Features**:
- **Matrix Evaluation**: Multiple models × multiple datasets
- **Caching**: S3-based result caching for collaboration
- **Dependency Tracking**: Automatic re-execution when code changes
- **Reproducibility**: Locked versions and shared artifacts

**DVC Commands**:
```bash
# Run full pipeline
dvc repro

# Force pipeline run (ignore cache)
dvc repro --force

# Run specific stage
dvc repro solve_sqa@claude-3.7

# Check pipeline status
dvc status

# Pull remote results
dvc pull

# Push local results  
dvc push
```

### **Testing Framework**
**Structure**: `tests/` directory

**Test Categories**:
```python
# Basic smoke test
def test_smoke_mockllm_arithmetic():
    inspect_ai.eval(
        "astabench/evals/demo/arithmetic/task.py",
        model="mockllm/model",
        solver=SolverSpec("agent_baselines/solvers/llm.py@llm_with_prompt"),
    )

# Expensive test (requires APIs)
@pytest.mark.expensive
def test_real_api_integration():
    # Tests with actual API calls
```

**Configuration**:
```toml
[tool.pytest.ini_options]
addopts = "-m \"not expensive\""  # Skip expensive tests by default
markers = ["expensive: mark test as expensive (skipped by default)"]
```

---

## ⚙️ **5. Configuration & Deployment**

### **Dependency Management** 
**File**: `pyproject.toml`

**Conflict Resolution**:
```toml
[tool.uv]
conflicts = [
    [{extra = "sqa"}, {extra = "storm"}],      # SQA conflicts with STORM
    [{extra = "sqa"}, {extra = "futurehouse"}], # SQA conflicts with FutureHouse  
    [{extra = "storm"}, {extra = "smolagents"}], # STORM conflicts with smolagents
]
```

**Solver-Specific Dependencies**:
```toml
[project.optional-dependencies]
sqa = ["ai2-scholar-qa==0.7.0"]
smolagents = ["smolagents==1.17.0"]
datavoyager = ["autogen-agentchat==0.6.4", "matplotlib==3.10.0"]
futurehouse = ["futurehouse-client==0.3.19"]
storm = ["knowledge-storm @ git+https://github.com/gituser768/storm.git@dh-fix-youcom"]
```

### **Environment Setup**

**Required Environment Variables**:
```bash
# Core API keys
export OPENAI_API_KEY=<your-openai-key>
export ANTHROPIC_API_KEY=<your-anthropic-key>  
export GOOGLE_API_KEY=<your-google-key>
export ASTA_TOOL_KEY=<your-asta-tool-key>
export HF_TOKEN=<your-huggingface-key>

# Solver-specific
export MODAL_TOKEN=<modal-token>        # For SQA solver
export MODAL_TOKEN_SECRET=<modal-secret> # For SQA solver
```

**Per-Solver Configuration**: `solvers/<name>/env` files
**Setup Scripts**: `solvers/<name>/setup.sh`

```bash
# Example setup.sh
#!/bin/bash
uv sync --extra sqa  # Install SQA-specific dependencies
```

### **Development Workflow**

**Adding New Solvers**:
1. Create `solvers/<name>/` directory with setup.sh, demo.sh, README.md
2. Implement solver in `agent_baselines/solvers/<name>/<name>.py`  
3. Add dependencies to pyproject.toml as optional extra
4. Update conflict resolution in pyproject.toml if needed
5. Add DVC pipeline stage if needed
6. Add tests in `tests/solvers/`

**Running Evaluations**:
```bash
# Individual solver testing
./solvers/react/demo.sh

# Full pipeline  
dvc repro

# Specific model/task
dvc repro solve_sqa@claude-3.7

# Docker environment
make shell SOLVER=sqa
export OPENAI_API_KEY=...
./solvers/sqa/demo.sh
```

**Local Development**:
```bash
# Install development dependencies
uv sync --extra dev

# Run tests
make test                    # Basic tests
make test-expensive          # API tests (requires keys)

# Code quality
make format                  # Format with black
make flake                   # Lint with flake8  
make mypy                    # Type check
```

---

## 📊 **6. Solver Comparison Matrix**

| Solver | Type | Dependencies | Key Features | Use Cases |
|--------|------|-------------|--------------|-----------|
| ReAct | General | astabench | Tool-calling loop, configurable tools | General problem solving |
| Smolagents | General | smolagents | Code generation, sandbox execution | Programming tasks |
| SQA | Specialized | ai2-scholar-qa | Literature retrieval, citation management | Academic Q&A |
| DataVoyager | Specialized | autogen | Multi-agent system, data analysis | Data exploration |
| Paper Finder | Specialized | - | Academic search, recommendations | Research discovery |
| Asta Tables | Specialized | - | Literature review tables | Systematic reviews |
| STORM | Research | knowledge-storm | Knowledge curation, report generation | Comprehensive reports |
| FutureHouse | Research | futurehouse-client | Scientific writing | Literature synthesis |

---

## 🎯 **7. Key Implementation Patterns**

### **Solver Interface Pattern**
```python
@solver
def my_solver(param1: str, param2: int = 10) -> Solver:
    """All solvers follow this pattern"""
    # Setup logic
    tools = get_tools_from_config()
    
    # Return Solver chain
    return chain([
        system_message("System prompt"),
        basic_agent(tools=tools, max_steps=param2)
    ])
```

### **Tool Integration Pattern**
```python
# Standard tool setup
toolset_config = ToolsetConfig(
    with_search_tools=1,
    with_table_editor=1,
    # ... other tool options
)
tools = toolset_config.create_tools(state)

# Add submission tool
submit_tool_instance = submit_tool(
    submit_name="submit",
    description="Submit final answer"
)
tools.append(submit_tool_instance)
```

### **State Management Pattern**
```python
# Merge tools with evaluation state
state = merge_tools_with_state(state, tools)

# Record model usage for evaluation
record_model_usage_with_inspect(usage_data, state)
```

### **Configuration Pattern**
```python
# Model normalization
model_name = normalize_model_name(completion_model)

# Environment-based configuration
modal_token = os.getenv("MODAL_TOKEN")
if not modal_token:
    raise ValueError("MODAL_TOKEN required for SQA solver")
```

---

## 🚀 **8. Advanced Topics**

### **Multi-Agent Systems (DataVoyager)**
- **Agent Roles**: Planner, Programmer, Analyst, Coordinator  
- **Communication**: YAML-configured message passing
- **Orchestration**: AutoGen framework integration
- **Tool Access**: Shared tool environment across agents

### **RAG Implementation (SQA)**
- **Retrieval**: PaperFinderWithReranker for academic papers
- **Reranking**: ModalReranker for relevance scoring  
- **Response Generation**: Template-based formatting with citations
- **Table Processing**: JSON to Markdown conversion with citation linking

### **Sandbox Execution (Smolagents)**
- **Code Generation**: LLM generates Python code for tasks
- **Sandbox Isolation**: Controlled execution environment
- **Tool Bridge**: Code can access InspectAI tools via wrapper functions
- **Error Handling**: Graceful failure and retry mechanisms

### **Evaluation Pipeline Architecture**
- **Matrix Evaluation**: Cartesian product of models × datasets × solvers
- **Artifact Management**: DVC tracks inputs, outputs, and intermediate results
- **Reproducibility**: Locked dependencies and versioned configurations
- **Collaboration**: Shared S3 cache for team development

---

## 🎯 **9. Key Takeaways**

### **Modular Architecture**
- Each solver is **self-contained** with its own dependencies
- **InspectAI interface** provides consistency across different agent types
- **Tool integration** through AstaBench standardizes capabilities

### **Evaluation-First Design**  
- **DVC pipeline** ensures reproducible comparisons
- **Matrix evaluation** across models and datasets
- **Caching system** for efficient collaboration

### **Practical Flexibility**
- **Docker isolation** handles complex dependency conflicts
- **Multiple tool formats** support different agent paradigms  
- **Configuration-driven** agent behavior and deployment

### **Research-Engineering Balance**
- **Academic rigor** in evaluation methodology
- **Engineering practices** for reliability and maintainability
- **Open source** approach for community contribution

This codebase demonstrates a sophisticated approach to AI agent evaluation, balancing research flexibility with engineering rigor. Each component serves the overarching goal of fair, reproducible agent comparison across diverse tasks and approaches.

---

## 📚 **10. Additional Resources**

- **AstaBench Documentation**: https://github.com/allenai/asta-bench
- **InspectAI Framework**: https://github.com/UKGovernmentBEIS/inspect_ai
- **DVC Documentation**: https://dvc.org/doc
- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/

---

*Last Updated: January 2025*
*Analysis Version: 1.0*