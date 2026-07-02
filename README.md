# nation_sim

A simple turned not-so-simple economic simulator modelling a "world" of countries that interact with tradeable resources, sovereign debt, demographic transitions, and autonomous agent strategies.

## Features

- **9 tradeable resources** (food, energy, metal, wood, goods, luxury, tech, fuel, services) accounting Engel's Law for proportional demand
- **3 income classes** (poor, middle, rich) tied to welfare
- **Dynamic pricing** for resources
- **Bilateral trade** with tariffs, distance-based trade costs, sovereign debt
- **Dynamic population growth** tied to welfare (peaks at intermediate welfare, declines at extremes)
- **Migration** toward higher-welfare countries, **tech growth**, **FX dynamics**, **capacity investment**
- **Crash events** (supply shocks, financial panics, droughts, energy crises, tech collapses)
- **4 scripted baseline agent** + **4 autonomous agent** profiles with dynamic tariff/trade strategies
- **Live Plotly/Dash dashboard** with stock exchange theming
- **CSV/Parquet logging**
- **Seed deterministic** generation.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python >= 3.10.

## Quick Start

```bash
# Run 240 months (ticks) with in-browser dashboard
sim
# or run it in the lame way with python -m
python -m nation_sim

# Set a custom tick count with the --ticks flag
sim --ticks 480
```
### Other commands

1. simulation w live dashboard

```bash
python -m nation_sim --ticks 240 --preset balanced --agents autonomous
```
you can pass these CLI arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--ticks` | `240` | Ticks/Months to simulate |
| `--preset` | `balanced` | World preset (`balanced`, `mercantilist`, `free_trade`) |
| `--agents` | `autonomous` | Agent strategy(`autonomous`, `buffer`, `growth`, `mercantilist`, `welfare`, `balanced`) |
| `--countries` | `10` | Number of countries |
| `--seed` | `0` | Set seed (0 random) |
| `--tick-delay-ms` | `200` | Delay between ticks (0 = full speed) |
| `--port` | `8050` | Dashboard port |
| `--no-dashboard` | off | No dashboard |
| `--no-browser` | off | No auto-open dashboard |
| `--parquet` | off | Write Parquet files |

2. quick compare all 8 policies on a set seed

```bash
sim-compare --ticks 120 --seed 42
```
you can use these CLI arguments:
| Flag | Default | Description |
|------|---------|-------------|
| `--ticks` | `120` | Ticks |
| `--seed` | `42` | Seed (set for standard comparison) |
| `--countries` | `10` | Countries |
| `--preset` | `balanced` | World preset |

3. Verify simulation (dev only)

```bash
pytest tests/
```

## Architecture

```
nation_sim/
├── __main__.py    # CLI entry point
├── core/          # Economics (production / consumption / market / currency / world state)
├── sim/           # Orchestration (engine tick loop / world generation / config)
├── agents/        # Policy implementations (scripted baselines / autonomous archetypes)
├── recording/     # Data recording
├── viz/           # Live Plotly/Dash dashboard
└── scripts/       # Additional CLI entry points
```

## Output

- `runs/latest/country.csv` — per-tick per-country metrics (welfare, population, reserves, GDP, performance index)
- `runs/latest/resource.csv` — per-tick per-country per-resource metrics (stockpile, price)
- Live dashboard at `http://127.0.0.1:8050`.
