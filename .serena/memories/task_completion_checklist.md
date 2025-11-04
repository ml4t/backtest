# Task Completion Checklist

## Before Considering a Task Complete

### 1. Code Quality Checks (MANDATORY)
```bash
make format      # Auto-format all code
make lint        # Fix any linting issues
make type-check  # Ensure type checking passes
# OR
make check       # Run all three at once
```

### 2. Run Tests
```bash
make test-qng    # Run qengine-specific tests
# OR
make test        # Run all tests if changes affect multiple projects
```

### 3. Verify Coverage
```bash
make test-cov    # Check test coverage (must be >80%)
```

### 4. Documentation
- Update docstrings if APIs changed
- Update README.md if user-facing changes
- Update FUNCTIONALITY_INVENTORY.md if new features added

### 5. Pre-commit Validation
```bash
pre-commit run --all-files  # If pre-commit hooks are installed
```

### 6. Review Changes
```bash
git status       # Review modified files
git diff         # Review actual changes
```

### 7. Stage and Commit (ONLY if explicitly requested)
```bash
git add -p       # Stage changes selectively
git commit -m "type: Description"  # Use conventional commit format
```

## Commit Message Format
- feat: New feature
- fix: Bug fix
- docs: Documentation only
- style: Code style changes
- refactor: Code refactoring
- test: Test additions/changes
- chore: Build process or auxiliary tool changes

## Important Reminders
- NEVER commit unless explicitly requested by user
- NEVER skip quality checks
- NEVER push without running tests
- Keep changes focused and atomic
- Update documentation alongside code changes

## Common Issues to Check
- [ ] No files in wrong directories (use .claude/ for planning)
- [ ] No temporary or work-in-progress files in root
- [ ] No commented-out code
- [ ] No print statements (use logging)
- [ ] No hardcoded paths or credentials
- [ ] All imports are used
- [ ] No circular dependencies