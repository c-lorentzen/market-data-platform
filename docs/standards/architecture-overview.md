# Architecture Overview (Project 0)

## Purpose
This document defines the high-level architecture for the Market Data Platform (Project 0 foundations).
The goal is to provide a modular, extensible base that supports future modules (ingestion, storage, features, backtesting,
ML, execution) without requiring refactoring of the core.

## Architectural Style
The system follows a layered architecture inspired by Clean Architecture:
- Domain (core): stable models, invariants, and interfaces (contracts)
- Application: use cases and orchestration (workflows)
- Infrastructure: external integrations (data sources, databases, logging implementations)
- Adapters: translation/mapping between external formats and domain types

Dependency direction is inward: Domain has no dependencies on other internal layers.

## Module Responsibilities (Conceptual)
### Domain
- Defines core types and stable interfaces (e.g., MarketDataSource, StorageRepository)
- Defines domain-level errors
- Remains framework-agnostic and pure where possible

### Application
- Orchestrates workflows via use cases (e.g., ingest -> validate -> store)
- Depends on domain interfaces, not concrete infrastructure
- Coordinates cross-module interaction without implementing external integrations

### Infrastructure
- Implements domain interfaces using concrete technologies (future MT5 adapter, SQL storage, etc.)
- Handles external exceptions and adapts them to domain/application-safe errors
- Provides logging implementations and environment-dependent services

### Adapters
- Map external payloads (API responses, DB rows) into domain models and vice versa
- Validate and normalize inputs at boundaries

## Extensibility Principle
New modules should be added by:
1) defining/using a domain interface (contract),
2) implementing it in infrastructure,
3) wiring it via bootstrap/config,
without changing domain code.

## Non-Goals (Project 0)
Project 0 does not implement trading logic, backtesting, ML models, or execution logic.
This phase defines the architecture and standards only.
