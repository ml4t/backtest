# Strategies guide

## Ownership

`templates.py` provides reusable strategy implementations on top of the public `Strategy`
lifecycle. Templates translate user signals and parameters into intents; they do not bypass broker,
execution, or risk policies.

## Constraints

Keep constructor defaults explicit and stable. A strategy may use only data available at the active
lifecycle phase, and generated intents must be deterministic for the same input stream. Test each
template through the public engine rather than testing its signal calculation in isolation.
