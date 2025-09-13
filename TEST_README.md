# Testing Guide for Paper Search Engine

This guide covers all testing aspects of the paper search engine implementation.

## 🧪 Test Structure

```
tests/
├── conftest.py              # Test configuration and fixtures
├── requirements.txt         # Test dependencies
├── unit/                   # Unit tests (fast, no external deps)
│   ├── test_bge_embedder.py
│   ├── test_document_chunker.py
│   ├── test_es_indexer.py
│   ├── test_ingest_papers.py
│   └── test_search_service.py
└── integration/            # Integration tests (require services)
    └── test_full_pipeline.py
```

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
pip install -r tests/requirements.txt
```

### 2. Run Tests

```bash
# Run all unit tests
make test-unit

# Run specific component tests
make test-embedder
make test-chunker
make test-indexer

# Run with custom script
python run_tests.py --type unit --verbose
```

## 📋 Test Categories

### Unit Tests ⚡
Fast tests with no external dependencies. Mock all external services.

```bash
# All unit tests
pytest tests/unit/

# Specific components
pytest tests/unit/test_bge_embedder.py -v
pytest tests/unit/test_document_chunker.py -v
pytest tests/unit/test_es_indexer.py -v
pytest tests/unit/test_ingest_papers.py -v
pytest tests/unit/test_search_service.py -v
```

### Integration Tests 🔗
Tests requiring running Elasticsearch and other services.

```bash
# Start services first
docker-compose up -d

# Run integration tests
pytest tests/integration/ -m integration -v

# Or use make
make test-integration
```

## 🎯 Test Markers

Use pytest markers to run specific test categories:

```bash
# Run only embedder tests
pytest -m embedder

# Run only fast tests
pytest -m "not slow"

# Run search-related tests
pytest -m search

# Available markers:
# - unit: Unit tests
# - integration: Integration tests
# - slow: Slow running tests
# - embedder: BGE embedder tests
# - chunker: Document chunker tests
# - indexer: Elasticsearch indexer tests
# - search: Search functionality tests
# - pipeline: Ingestion pipeline tests
```

## 🛠 Test Tools & Scripts

### Test Runner Script

```bash
# Interactive test runner
python run_tests.py

# Specific options
python run_tests.py --type unit --verbose --coverage
python run_tests.py --type component --component embedder
python run_tests.py --type specific --pattern "test_search"
python run_tests.py --type integration
```

### Makefile Commands

```bash
make test           # Run all tests
make test-unit      # Unit tests only
make test-integration  # Integration tests
make test-fast      # Exclude slow tests
make test-coverage  # With coverage report
make test-parallel  # Parallel execution
```

## 📊 Coverage Reports

Generate coverage reports to see test coverage:

```bash
# HTML coverage report
pytest --cov=data_pipeline --cov=backend --cov-report=html

# Terminal coverage
pytest --cov=data_pipeline --cov=backend --cov-report=term

# Using make
make test-coverage
```

View HTML report: `open htmlcov/index.html`

## 🔧 Test Configuration

### pytest.ini
Main pytest configuration with markers, output settings, and test discovery rules.

### conftest.py
Shared fixtures and test utilities:
- Mock components (BGE embedder, ES client)
- Sample test data
- Temporary directories
- Test configuration

## 🧪 Test Examples

### Unit Test Example

```python
def test_bge_embedder_encode():
    """Test BGE embedder encoding"""
    embedder = BGEEmbedder()

    text = "Test document"
    embedding = embedder.encode(text)

    assert embedding.shape == (1024,)
    assert not np.isnan(embedding).any()
```

### Integration Test Example

```python
@pytest.mark.integration
def test_full_pipeline():
    """Test complete ingestion pipeline"""
    # Requires running ES and MinIO
    processor = PaperProcessor()
    search_service = SearchService()

    # Process and search
    processor.ingest_directory("test_papers/")
    results = search_service.search("transformer")

    assert len(results) > 0
```

### Mock Usage Example

```python
@patch('data_pipeline.bge_embedder.AutoModel.from_pretrained')
def test_with_mock(mock_model):
    """Test with mocked transformer model"""
    mock_model.return_value = Mock()

    embedder = BGEEmbedder()
    # Test logic here
```

## 🏃‍♂️ Running Tests

### Local Development

1. **Quick unit tests** (development cycle):
   ```bash
   make test-unit
   ```

2. **Full test suite** (before commits):
   ```bash
   make test-all
   ```

3. **Component-specific testing**:
   ```bash
   python run_tests.py --type component --component search --verbose
   ```

### CI/CD Pipeline

```bash
# Install dependencies
pip install -r requirements.txt -r tests/requirements.txt

# Run linting
make lint

# Run unit tests with coverage
make test-coverage

# Run integration tests (if services available)
make test-integration
```

## 🐛 Debugging Tests

### Verbose Output
```bash
pytest -v -s  # -s shows print statements
```

### Debug Specific Test
```bash
pytest tests/unit/test_embedder.py::TestBGEEmbedder::test_encode -v -s
```

### Use pdb for debugging
```python
def test_debug_example():
    import pdb; pdb.set_trace()
    # Your test code
```

### Check Test Dependencies
```bash
python run_tests.py --type unit --no-deps-check
```

## 📈 Performance Testing

### Benchmark Tests
```bash
pytest tests/ -m benchmark --benchmark-only
```

### Timing Tests
```bash
pytest --durations=10  # Show 10 slowest tests
```

## 🔄 Test Data Management

### Sample Data Creation
Tests use fixtures in `conftest.py` to create:
- Sample markdown files
- Mock embeddings
- Temporary directories
- Test configurations

### Cleanup
Tests automatically clean up temporary files and test indices.

## ⚠️ Common Issues

### Elasticsearch Connection
```bash
# Check ES is running
curl http://localhost:9202/_cluster/health

# Restart if needed
docker-compose restart paper-search-elasticsearch
```

### Import Errors
```bash
# Check Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Or install in development mode
pip install -e .
```

### Mock Issues
- Ensure mocks are patched at the right location
- Use `spec=` parameter for better mocking
- Check patch target path

### Slow Tests
- Use `-m "not slow"` to skip slow tests
- Run integration tests separately
- Use parallel execution: `pytest -n auto`

## 📝 Writing New Tests

### Unit Test Guidelines
1. Test single components in isolation
2. Mock all external dependencies
3. Use descriptive test names
4. Test both success and error cases
5. Keep tests fast (< 100ms each)

### Integration Test Guidelines
1. Test component interactions
2. Use real services when possible
3. Include cleanup in fixtures
4. Test realistic data flows
5. Mark with `@pytest.mark.integration`

### Test File Structure
```python
"""
Module docstring explaining what's tested
"""

import pytest
from unittest.mock import Mock, patch

class TestComponentName:
    """Test cases for ComponentName"""

    @pytest.fixture
    def component_instance(self):
        """Fixture for component setup"""
        return ComponentName()

    def test_basic_functionality(self, component_instance):
        """Test basic component functionality"""
        # Arrange
        input_data = "test input"

        # Act
        result = component_instance.process(input_data)

        # Assert
        assert result is not None
        assert isinstance(result, expected_type)
```

## 🎉 Best Practices

1. **Run tests frequently** during development
2. **Write tests first** for new features (TDD)
3. **Mock external dependencies** in unit tests
4. **Use descriptive test names** that explain what's tested
5. **Keep tests independent** - no test should depend on another
6. **Test edge cases** and error conditions
7. **Use fixtures** for common test setup
8. **Clean up** test data and resources
9. **Run full test suite** before committing
10. **Monitor test coverage** and aim for >80%