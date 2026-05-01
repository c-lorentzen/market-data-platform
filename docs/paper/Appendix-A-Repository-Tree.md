# Appendix A — Repository Tree (Project 0)

This appendix documents the repository structure established in Project 0.

```text
market-data-platform/
├─ README.md
├─ docs/
│  ├─ adr/
│  │  ├─ ADR-0001-python-first.md
│  │  ├─ ADR-0002-clean-architecture.md
│  │  ├─ ADR-0003-configuration-strategy.md
│  │  ├─ ADR-0004-logging-policy.md
│  │  └─ ADR-0005-testing-strategy.md
│  ├─ diagrams/
│  ├─ paper/
│  │  ├─ Project0_Charter.md
│  │  └─ Appendix-A-Repository-Tree.md
│  └─ standards/
│     ├─ project0-scope.md
│     ├─ architecture-overview.md
│     ├─ dependency-rules.md
│     └─ standards-index.md
│
├─ scripts/
│  ├── db_ping.py
│  ├── create_tables.py
│  ├── insert_sample_candles.py
│  ├── mt5_ping.py
│  ├──ingest_mt5_rates.py
│  └── fetch_step_index_rates.py
│
├─ src/
│  └── mdp/
│     ├── database/
│     │   ├── __init__.py
│     │   ├── connection.py
│     │   ├── schema.py
│     │   ├── schema.py
│     │   └── candle_repository.py
│     │
│     └── ingestion/
│         ├── __init__.py
│         └── mt5_connector.py
│
└─ tests/
   ├─ unit/
   ├─ integration/
   └─ contract/
