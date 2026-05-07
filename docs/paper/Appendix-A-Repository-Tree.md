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
│  ├── ingest_mt5_rates.py
│  ├── fetch_step_index_rates.py
│  ├── plot_step_index_close.py
│  ├── plot_market_candles.py
│  ├── plot_market_close.py 
│  ├── check_candle_quality.py
│  ├── build_basic_features.py
│  ├── analyze_market_structure.py
│  ├── analyze_multi_timeframe_structure.py
│  ├──  inspect_timeframe_alignment.py
│  ├── validate_timeframe_aggregation.py
│  ├── validate_timeframe_aggregation_batch.py
│  ├── plot_market_levels.py
│  ├── detect_support_resistance.py
│  ├── evaluate_support_resistance.py
│  ├── analyze_zone_context.py
│  ├── test_candle_loader.py
│  ├── walk_forward_sr_bounce.py
│  ├── generate_walk_forward_pdf_report.py
│  ├── detect_trend_channels.py
│  ├── plot_trend_channels.py
│  ├── detect_rolling_trend_channels.py
│  └── plot_rolling_trend_channels.py
│
├─ src/
│  └── mdp/
│     ├── database/
│     │   ├── __init__.py
│     │   ├── connection.py
│     │   ├── schema.py
│     │   ├── schema.py
│     │   ├── candle_repository.py
│     │   └── candle_loader.py
│     │
│     ├── ingestion/
│     │   ├── __init__.py
│     │   └── mt5_connector.py
│     │
│     ├── visualization/
│     │   ├── __init__.py
│     │   ├── price_charts.py
│     │   ├── trend_channel_charts.py
│     │   └── rolling_trend_channel_charts.py
│     │
│     ├── quality/
│     │   ├── __init__.py
│     │   └── candle_quality.py
│     │ 
│     ├── features/
│     │   ├── __init__.py
│     │   └── basic_features.py
│     │
│     ├── structure/
│     │   ├── __init__.py
│     │   ├── market_structure.py
│     │   ├── support_resistance.py
│     │   ├── zone_validation.py
│     │   ├── zone_context.py
│     │   ├── trend_channels.py
│     │   └── rolling_trend_channels.py
│     │
│     ├── timeframes/
│     │   ├── __init__.py
│     │   ├── alignment.py
│     │   └── registry.py
│     │
│     ├── utils/
│     │   ├── __init__.py
│     │   └── date_ranges.py
│     │
│     ├── backtesting/
│     │   ├── __init__.py
│     │   ├── engine.py
│     │   ├── walk_forward.py
│     │   └── performance.py
│     │
│     └── reporting/
│         ├── __init__.py
│         ├── walk_forward_exports.py
│         ├── performance_charts.py
│         └── pdf_report.py
│       
└─ tests/
   ├─ unit/
   ├─ integration/
   └─ contract/
