# ml4t.backtest Development Commands

## Essential Development Commands

### Code Quality (ALWAYS run before committing)
```bash
make format      # Auto-format code with ruff
make lint        # Check code style with ruff
make type-check  # Validate types with mypy
make check       # All three above combined
make quality     # Format, lint, type-check all projects in monorepo
```

### Testing
```bash
make test        # Run all tests
make test-qng    # Test ml4t.backtest project specifically
make test-unit   # Unit tests only
make test-cov    # With coverage report
pytest tests/unit/test_specific.py -v  # Run specific test file
pytest tests/unit/test_file.py::TestClass::test_method -xvs  # Run specific test
```

### Development Setup
```bash
make dev-install # Install pre-commit hooks
make setup       # One-time monorepo setup (from parent dir)
pre-commit run --all-files  # Manual pre-commit check
```

### Git Commands
```bash
git status       # Check changes
git add -p       # Stage selectively
git commit -m "feat: Description"  # Commit with conventional message
git push         # Push after all checks pass
```

### System Commands (Linux)
```bash
ls -la          # List files with details
cd directory    # Change directory
grep -r "pattern" .  # Search for pattern
find . -name "*.py"  # Find Python files
```

### Python/Project Running
```bash
python -m pytest  # Run tests
python examples/benchmark.py  # Run example scripts
python -m cProfile -o profile.stats examples/benchmark.py  # Profile code
```

## Important Notes
- NEVER commit without running `make check` first
- This is part of a monorepo - some commands should be run from parent directory
- Use `make test-qng` for ml4t.backtest-specific tests
- Always follow TDD: write test first, then implementation