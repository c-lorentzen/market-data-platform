# Dependency Rules (Project 0)

## Core Rule
Dependencies must flow inward. Inner layers must not depend on outer layers.

## Allowed Imports
- Domain: may not import from Application/Infrastructure/Adapters.
- Application: may import Domain; may not import Infrastructure directly.
- Infrastructure: may import Domain and Application (to implement interfaces and wire services).
- Adapters: may import Domain; may be used by Infrastructure/Application to translate data at boundaries.

## Practical Guidelines
- Domain contains interfaces/contracts; infrastructure provides implementations.
- No module should read environment variables directly except a centralized bootstrap/settings layer.
- External libraries (MT5 SDK, DB drivers) must be isolated to Infrastructure.
- Conversion/parsing logic belongs in Adapters, not Domain.

## Rationale
These rules preserve modularity, testability, and replaceable infrastructure, enabling future expansion without refactoring core.
