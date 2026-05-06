# Market Data Platform - Project State

## Project Owner

Cristhian Guido Lorentzen Burgos

International student at Santa Fe College studying Programming and Analysis. Holds a B.S. in Industrial Engineering and is building this project as a serious Data Science / Quantitative Trading portfolio project.

## Project Goal

Build a modular, scalable, data-driven market analysis and backtesting system using Python, Pandas, NumPy, MySQL, MT5 data, and later machine learning.

The system should work like Lego blocks, where each module can operate independently and later combine with other modules.

## Current Local Path

D:\trading-project\market-data-platform

## GitHub Remote

https://github.com/c-lorentzen/market-data-platform.git

The repository is private.

## Core Principles

- Build step by step.
- Avoid unnecessary complexity.
- Use mathematical definitions, not subjective trading language.
- Keep modules independent and reusable.
- Separate strategy logic from the backtesting engine.
- Avoid look-ahead bias.
- Use walk-forward validation for serious testing.
- Use Git commits after important milestones.
- Never commit `.env`, passwords, generated outputs, or local virtual environments.

## Coding Style Requirement

All future code should include concise English student-style comments inside the code, not only docstrings.

Example:

```python
# Ignore upcoming signals while the previous trade is still open.
if signal_time <= next_allowed_entry_time:
    continue
Comments should explain each important block, function call, and logic section.

Main Technologies
Python
Pandas
NumPy
MySQL
MetaTrader 5 Python API
Matplotlib
mplfinance
ReportLab
Git / GitHub
Current Project Stage

The project has moved beyond basic setup and is now entering systematic strategy design.

Current active stage:

Trend Channel Detection v1

Completed Modules
1. Database and MySQL

The project has a MySQL database named market_data_platform.

Main table:

market_candles

The table stores OHLCV data from MT5.

Database connection uses .env.

Important rule:

.env must never be committed.

2. MT5 Data Ingestion

MT5 connection works with Deriv synthetic markets.

Confirmed server:

DerivBVI-Server

Confirmed symbols include:

Step Index
Volatility 10 Index
Volatility 25 Index
Volatility 50 Index
Volatility 75 Index
Volatility 100 Index

Data ingestion works by:

candle count
date range
aligned multi-timeframe parent/child windows
3. Multi-Timeframe Alignment

The system can validate that child candles reconstruct parent candles.

Example:

1 H1 candle = 4 M15 candles
1 H1 candle = 12 M5 candles
1 H1 candle = 60 M1 candles

Validation has reached:

time alignment rate: 100%
OHLC match rate: 100%
4. Flexible Candle Loading

The project supports loading candles by:

--limit
--start-time and --end-time
--lookback-days
--lookback-weeks
--lookback-months

This is handled by:

src/mdp/utils/date_ranges.py
src/mdp/database/candle_loader.py
5. Data Quality and Features

The project includes:

candle quality checks
basic candle features
rolling volatility
rolling highs/lows
basic market structure classification
multi-timeframe structure analysis
6. Support and Resistance

The system detects support/resistance as zones, not just exact price lines.

Features include:

auto zone width
tolerance percent
historical respect rate
break rate
zone context relative to current price
support/resistance visualization
7. Backtesting Engine

The core backtesting engine is strategy-agnostic.

Important files:

src/mdp/backtesting/engine.py

The engine supports:

long trades
short trades
entry price
stop loss
take profit
win
loss
time_exit
ambiguous
invalid
overlapping trade control
R multiple calculation
8. Strategy Plugin: Support/Resistance Bounce

Important file:

src/mdp/strategies/support_resistance_bounce.py

This is a baseline strategy only. It is not the final strategy.

Features:

support/resistance bounce logic
minimum respect rate
minimum RR
minimum risk points
confirmation mode:
touch
rejection_close

The baseline strategy performed poorly in walk-forward validation, which is useful because it proved that the system can measure weak strategies honestly.

9. Trade Visualization

Important file:

src/mdp/visualization/backtest_charts.py

The system can generate TradingView-style visualizations with:

candlesticks
support/resistance zones
entry markers
exit markers
SL lines
TP lines
reward/risk rectangles
trade outcome labels
10. Walk-Forward Validation

Important file:

src/mdp/backtesting/walk_forward.py

The system supports generic walk-forward validation:

training window
testing window
step size
strategy-injected signal builder

This reduces look-ahead bias by detecting conditions on past data and testing signals on future data.

11. Performance Metrics

Important file:

src/mdp/backtesting/performance.py

Metrics currently include:

equity curve
drawdown
profit factor
expectancy
average win
average loss
payoff ratio
profitable rate
TP/SL win rate
Sharpe-like R metric
trade distribution
win/loss streaks
12. Reporting

Important files:

src/mdp/reporting/walk_forward_exports.py
src/mdp/reporting/performance_charts.py
src/mdp/reporting/pdf_report.py
scripts/generate_walk_forward_pdf_report.py

The system can export:

windows.csv
signals.csv
trades.csv
summary.csv
config.json
performance_summary.csv
equity_curve.csv
trade_distribution.csv
walk_forward_report.pdf

The PDF is functional but not yet visually polished.

Current Strategy Direction

The support/resistance bounce strategy is only a baseline.

The real target strategy is based on trend channels and multi-timeframe analysis.

The discretionary idea:

D1 defines macro bias.
H4/H1 define internal structure.
M15/M5/M1 refine precise entries.
Trend channels are used to detect structured movement.
Main channels and secondary channels may exist.
Broken channels may act as dynamic support/resistance.
The strategy should identify breakouts, retests, fake breakouts, and explosive moves.
Trend Channel Detection v1

Recently added files:

src/mdp/structure/trend_channels.py
scripts/detect_trend_channels.py

Current capabilities:

prepare OHLC candle data
detect swing pivots using left/right windows
fit regression lines to pivot highs/lows
build ascending and descending channel candidates
calculate channel width using envelope quantile
calculate containment rate
count upper and lower channel touches
classify current price state:
inside_channel
near_upper_boundary
near_lower_boundary
above_channel
below_channel
calculate deterministic quality_score
Recent Trend Channel Test Results
Step Index M1, limit 300

Detected one ascending channel.

Important interpretation:

The detector works.
The current price was inside the detected channel.
Containment rate was acceptable.
R-squared was low, meaning the pivot line fit was not very clean.
This is okay for v1, but visual validation is needed.
Step Index M1, limit 1000

Detected no global channel.

Important interpretation:

A single global channel across 1000 M1 candles is too broad.
The market likely contains multiple regimes.
This confirms the need for rolling/segmented channel detection.
Step Index H1, limit 300

Detected one descending channel.

Important interpretation:

The detector can identify macro channel candidates.
The channel had high containment but low R-squared.
Visual validation is needed before using channels in a strategy.
Current Conclusion

Trend Channel Detection v1 works as an initial mathematical detector.

However, it currently detects a simple global channel candidate.

The next step should not be a strategy yet.

The next step should be visualization.

Next Recommended Step

Create Trend Channel Visualization v1.

Suggested files:

src/mdp/visualization/trend_channel_charts.py
scripts/plot_trend_channels.py

The chart should show:

candlesticks
pivot highs
pivot lows
detected lower channel line
detected upper channel line
channel type
quality score
containment rate
current price state

Purpose:

Visually validate whether the mathematical channel detector matches what a discretionary trader would draw manually.

After Visualization

Next improvements should be:

Rolling/segmented channel detection.
Detect primary and secondary channels.
Rank channels by quality score.
Penalize channels that are too wide.
Penalize low R-squared channels.
Detect channel breakout.
Detect breakout retest.
Detect false breakout.
Create Trend Channel Breakout/Retest Strategy v1.
Run backtesting and walk-forward validation on that strategy.
Important Future Strategy Rules

All strategy rules must be mathematical and based on OHLC data.

No subjective phrases should be used in strategy implementation.

Everything should be measurable.

Examples:

trend slope
pivot sequence
channel containment rate
breakout distance
close outside channel
close back inside channel within N candles
wick rejection ratio
rolling volatility expansion
momentum continuation
RR threshold
walk-forward win probability proxy
Do Not Do Yet
Do not jump into SMC yet.
Do not add machine learning yet.
Do not create auto-trading execution yet.
Do not overcomplicate the strategy before validating the channel detector.
Do not optimize parameters before having a stable strategy definition.
Current Next Command Idea

Continue from:

Trend Channel Visualization v1

Then test:

python scripts\plot_trend_channels.py --symbol "Step Index" --timeframe M1 --limit 300

and visually inspect whether the detected channels make sense.


---

# Paso 3 — Guarda el archivo

En Notepad:

```text
File → Save