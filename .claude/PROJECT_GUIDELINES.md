# QEngine Project Guidelines

## Repository Organization

### Directory Structure

```
qengine/
├── .claude/              # Claude AI planning and reference docs
│   ├── planning/         # Implementation plans, roadmaps
│   ├── reference/        # Design reviews, decisions
│   └── PROJECT_GUIDELINES.md
├── docs/                 # User-facing documentation
│   ├── architecture/     # System design docs
│   ├── guides/          # How-to guides
│   └── api/             # API reference
├── src/                 # Source code
│   └── qengine/
├── tests/               # Test suite
├── examples/            # Example strategies
├── benchmarks/          # Performance benchmarks
└── README.md           # Project overview (user-facing)
```

### Document Placement Rules

1. **Root Directory**: Keep minimal
   - README.md (user-facing overview)
   - LICENSE
   - pyproject.toml
   - Makefile
   - .pre-commit-config.yaml
   - NO planning documents
   - NO work-in-progress files

2. **.claude/ Directory**: AI assistant workspace
   - Implementation plans
   - Design decisions
   - Work-in-progress notes
   - Review feedback
   - Internal roadmaps

3. **docs/ Directory**: User documentation
   - Architecture descriptions
   - Migration guides
   - API documentation
   - Tutorials

4. **Never Commit**:
   - Temporary files
   - Personal notes
   - Debug outputs
   - Cache directories

## Code Quality Standards

### Pre-commit Hooks

All code must pass pre-commit hooks before committing:

```bash
# Install hooks
pre-commit install

# Run manually
make pre-commit

# Auto-fix issues
make format
```

### Code Hygiene Tools

1. **Ruff**: Fast Python linter and formatter
   - Configuration in pyproject.toml
   - Run: `make format` and `make lint`

2. **MyPy**: Static type checking
   - Strict mode enabled
   - Run: `make type-check`

3. **Pytest**: Testing framework
   - Minimum 80% coverage
   - Run: `make test`

4. **Bandit**: Security checking
   - Automated via pre-commit
   - Manual: `bandit -r src/`

### Python Style Guide

1. **Type Hints**: Required for all public APIs
```python
def process_event(event: Event, timestamp: datetime) -> Optional[Order]:
    ...
```

2. **Docstrings**: Google style for all public functions
```python
def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate simple returns from price series.

    Args:
        prices: Array of prices

    Returns:
        Array of returns

    Raises:
        ValueError: If prices array is empty
    """
```

3. **Imports**: Organized in groups
```python
# Standard library
import os
from datetime import datetime

# Third party
import numpy as np
import polars as pl

# Local
from qengine.core import Event
```

## Development Workflow

### Before Starting Work

1. Check current state
```bash
git status
make clean
```

2. Update dependencies
```bash
pip install -e ".[dev]"
pre-commit autoupdate
```

### During Development

1. Use meaningful branch names
```bash
git checkout -b feature/add-market-impact-model
git checkout -b fix/order-execution-bug
git checkout -b docs/improve-migration-guide
```

2. Run checks frequently
```bash
make check  # Format, lint, type-check
make test   # Run tests
```

3. Keep commits atomic and well-described
```bash
git add -p  # Stage selectively
git commit -m "feat: Add Almgren-Chriss market impact model

- Implement square-root impact function
- Add configuration parameters
- Include unit tests"
```

### Before Committing

1. Run full quality assurance
```bash
make qa  # All checks + tests
```

2. Update documentation if needed
3. Ensure no files in wrong directories

### Code Review Checklist

- [ ] Code passes all automated checks
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No files in root that belong in .claude/
- [ ] No debug code or print statements
- [ ] Type hints for public APIs
- [ ] Docstrings for public functions

## Testing Standards

### Test Organization

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
├── scenarios/      # Golden scenario tests
└── benchmarks/     # Performance tests
```

### Test Requirements

1. Every new feature needs tests
2. Every bug fix needs a regression test
3. Maintain >80% code coverage
4. Use property-based testing for algorithms

### Test Naming

```python
def test_event_bus_delivers_events_in_chronological_order():
    """Clear, descriptive test names."""
    pass

def test_pit_data_prevents_future_access():
    """Test names describe expected behavior."""
    pass
```

## Documentation Standards

### User Documentation

1. Keep in `docs/` directory
2. Write for the target audience
3. Include code examples
4. Update with every feature

### Internal Documentation

1. Keep in `.claude/` directory
2. Record design decisions
3. Document trade-offs
4. Track technical debt

### Code Comments

1. Explain WHY, not WHAT
```python
# Bad
x = x + 1  # Increment x

# Good
x = x + 1  # Account for 1-based indexing in market data
```

2. Document complex algorithms
3. Add TODO with issue numbers
```python
# TODO(#123): Optimize this loop with Numba
```

## Performance Guidelines

### Profiling

Before optimizing:
```bash
make profile
py-spy record -o profile.svg -- python examples/benchmark.py
```

### Optimization Priority

1. Algorithmic complexity first
2. Data structure choices second
3. Micro-optimizations last

### Performance Tracking

Document benchmarks in `.claude/benchmarks/`:
- Baseline measurements
- Optimization attempts
- Results and trade-offs

## Security Considerations

### Never Commit

- API keys or secrets
- Personal data
- Proprietary algorithms
- Production configurations

### Use Environment Variables

```python
import os
api_key = os.environ.get('BROKER_API_KEY')
```

### Security Scanning

Automated via pre-commit:
- Bandit for Python code
- Secret scanning
- Dependency vulnerability checks

## Continuous Improvement

### Regular Maintenance

Weekly:
- Update dependencies
- Run full test suite
- Review open issues

Monthly:
- Performance profiling
- Documentation review
- Dependency audit

### Technical Debt

Track in `.claude/planning/TECH_DEBT.md`:
- What needs refactoring
- Why it was deferred
- Impact assessment
- Priority level

## Summary

Remember:
1. **Keep root clean** - Planning in .claude/, user docs in docs/
2. **Maintain quality** - Use pre-commit hooks and make commands
3. **Test everything** - Minimum 80% coverage
4. **Document clearly** - Both code and decisions
5. **Profile first** - Measure before optimizing

These guidelines ensure QEngine remains maintainable, performant, and professional.
