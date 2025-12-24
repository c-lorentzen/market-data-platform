# Project 0 — Scope and Constraints (Foundations)

## Purpose
Project 0 defines the foundational architecture for a modular, extensible, and testable data-driven financial system.
This phase establishes the repository structure, architectural layers, standards, and documentation that enable future modules
(e.g., ingestion, storage, feature engineering, backtesting, ML, execution) to be added without refactoring the core.

## Research Question
How can a scalable and modular software architecture be designed to support extensible, data-driven financial systems
while ensuring testability, maintainability, and future expansion?

## In Scope (What Project 0 WILL deliver)
- A Git-ready repository skeleton (docs/, src/, tests/)
- Defined architectural layers and responsibilities (e.g., domain/application/infrastructure)
- Defined module boundaries and dependency rules (who can import who)
- Interface definitions (“contracts”) for future modules (conceptual and/or stubbed)
- Coding standards and naming conventions
- Configuration and environment management philosophy (design + example)
- Logging and error-handling philosophy (design + example)
- Testing strategy (unit/integration/contract) and seed tests that validate architecture assumptions
- Architecture diagrams (context + layers + module/plugin overview)

## Out of Scope (What Project 0 WILL NOT include)
- No trading strategies implemented
- No machine learning models implemented
- No backtesting engine implemented
- No automated execution / live trading logic implemented
- No performance benchmarking of trading algorithms
- No real MT5 integration or SQL integration beyond conceptual interfaces/stubs

## Assumptions
- Python-first implementation strategy
- Future modules may vary in technology (e.g., different storage engines), so core must remain stable
- Architecture must support both research workflows (experimentation) and production-quality practices (testing, logging, CI-ready)

## Acceptance Criteria (Definition of Done)
Project 0 is complete when:
1. Repository structure exists and is committed to GitHub.
2. Architecture documentation clearly defines layers, responsibilities, and dependency direction.
3. Interface contracts are documented (and optionally stubbed in code) for future ingestion/storage/backtesting/strategy/ML/execution.
4. Configuration policy is documented and an example configuration is provided.
5. Logging and error-handling policy is documented and an example is provided.
6. Testing strategy is documented and at least minimal seed tests exist (architecture-level tests).
7. Architecture diagrams are included in the repository.
8. APA 7 technical paper draft exists and references the repository artifacts via appendices.

## Non-Goals (Explicit)
- This phase is not intended to produce profitable trading performance.
- This phase is not intended to validate strategies or predictive models.
- This phase is focused on maintainable software architecture and documentation.

## Future Phases (High-Level Roadmap)
- Project 1: Market data ingestion module (MT5 or alternative source)
- Project 2: Storage integration (SQL repository + migrations)
- Project 3: Feature engineering pipeline
- Project 4: Backtesting engine integration
- Project 5: ML experimentation and model lifecycle design
- Project 6: Execution, monitoring, and risk controls