## How to Run

### Prerequisites

- Python 3.12 or higher
- UV package manager (recommended) or pip

### Setup and Installation

#### Option 1: Using UV (Recommended)

1. Install UV (if not already installed) by following the [instructions at the official website](https://docs.astral.sh/uv/getting-started/installation/).

2. Create virtual environment and install dependencies:

   ```bash
   uv sync
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

#### Option 2: Using pip

1. Navigate to the project root directory.

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation

#### Full experiment pipeline (time-consuming)

To reproduce the complete set of experiments reported in the paper, we provide a bash script that executes all agents, networks, scenarios, and horizons sequentially. This pipeline is computationally expensive and can take several hours to complete.

For this reason, we recommend running only a single agent or configuration for testing and inspection purposes.

If you still wish to run the full pipeline:

1. Navigate to the project [`src/`](src/) directory.

2. Execute the script:

```bash
# Make the script executable
chmod +x scripts/run_paper_matrix.sh
# Run the script
./scripts/run_paper_matrix.sh phaseA # For 24h experiments
./scripts/run_paper_matrix.sh phaseB # For 168h experiments
```

#### Run individual agents (recommended for testing)

To quickly test the system or inspect the behavior of a specific agent, it is preferable to run individual agent scripts rather than the full experiment matrix.

```bash
python -m ml.bandit_contextual_rh --config configs/network_porto20_s600.yaml --hours 24 --episodes 10 --scenario baseline --out out/test.json
```

## Results aggregation and plots (analyze_results.py)

After running the experiment matrix, all raw outputs are stored as JSON files under:

- `out/paper/phaseA_24h/<network>/<method>/<scenario>.json`
- `out/paper/phaseB_168h/<network>/<method>/<scenario>.json`

To aggregate these runs into a single table and generate plots, use `src/scripts/analyze_results.py`.
This script scans a root folder (e.g., `out/paper/phaseA_24h`) and produces:

- `out/analysis/master_runs.csv` — one row per (phase, network, method, scenario) with mean±std KPIs
- `out/analysis/robustness.csv` — scenario deltas vs baseline for the same (phase, network, hours, method)
- `out/analysis/plots/*.png` — summary figures (J tables, stacked objective decomposition, and bandit arm-pulls when available)

### Run analysis for the 24h experiments

From the `src/` directory:

```bash
python -m scripts.analyze_results \
  --root out/paper/phaseA_24h \
  --out out/analysis/phaseA_24h
```

### Run analysis for the 128h experiments

```bash
python -m scripts.analyze_results \
  --root out/paper/phaseB_168h \
  --out out/analysis/phaseB_168h
```
