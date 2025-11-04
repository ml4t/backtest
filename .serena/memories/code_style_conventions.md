# QEngine Code Style and Conventions

## Code Style
- **Formatter**: ruff (auto-formats on `make format`)
- **Linter**: ruff (checks on `make lint`)
- **Type Checker**: mypy (strict typing required)
- **Python Version**: 3.12+ with modern features

## Naming Conventions
- **Classes**: PascalCase (e.g., `BacktestEngine`, `DataFeed`)
- **Functions/Methods**: snake_case (e.g., `get_next_event`, `on_market_event`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_EVENTS`)
- **Private methods**: Leading underscore (e.g., `_setup_event_handlers`)
- **Module files**: snake_case (e.g., `event_bus.py`)

## Type Hints
- **MANDATORY**: All functions must have type hints
- **Style**: Use modern Python 3.12+ syntax
  ```python
  def process_events(events: list[Event]) -> dict[str, Any]:
      ...
  ```
- **Optional types**: Use `Optional[T]` or `T | None`
- **Imports**: From `typing` module or built-in types

## Docstrings
- **Format**: Google style docstrings
- **Required for**: All public classes, methods, and functions
- **Structure**:
  ```python
  def calculate_returns(prices: np.ndarray, method: str = "simple") -> np.ndarray:
      """Calculate returns from price series.
      
      Args:
          prices: Array of prices
          method: Calculation method ('simple' or 'log')
      
      Returns:
          Array of returns
      
      Raises:
          ValueError: If prices array is empty
      """
  ```

## Testing Conventions
- **TDD Required**: Write test FIRST, then implementation
- **Test Location**: Mirror source structure in `tests/`
- **Test Naming**: `test_<functionality>` functions
- **Fixtures**: Use pytest fixtures from `conftest.py`
- **Coverage**: Minimum 80% required
- **Mocking**: Mock external dependencies

## Import Organization
1. Standard library imports
2. Third-party imports
3. Local imports
(Groups separated by blank lines)

## File Organization
- Keep modules focused and single-purpose
- Use `__init__.py` for public API exposure
- Define `__all__` for explicit exports

## Comments
- **Avoid unless necessary**: Code should be self-documenting
- **When needed**: Explain WHY, not WHAT
- **Never add comments unless explicitly requested**

## Error Handling
- Use specific exceptions
- Provide helpful error messages
- Document raised exceptions in docstrings

## Performance Considerations
1. Algorithmic complexity first
2. Polars query optimization
3. Numba JIT for tight loops
4. Profile before optimizing