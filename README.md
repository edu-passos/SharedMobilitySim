# M.IA - Modelling and Simulation - 2025/2026

> Curricular Unit: MS - [Modelação e Simulação](https://sigarra.up.pt/feup/pt/UCURR_GERAL.FICHA_UC_VIEW?pv_ocorrencia_id=560076)

## 1st Year - 1st Semester - 1st Project - Management of Shared Mobility (Topic 7)

## Group AI20

- Eduardo Passos - E-mail: up202205630@up.pt
- Guilherme Silva - E-mail: up202205298@up.pt
- Valentina Cadime - E-mail: up202206262@up.pt

## How to Run

### Prerequisites

- Python 3.12 or higher
- UV package manager (recommended) or pip

### Setup and Installation

#### Option 1: Using UV (Recommended)

1. Install UV (if not already installed) by following the [instructions at the official website](https://docs.astral.sh/uv/getting-started/installation/).

2. Install dependencies:

   ```bash
   uv sync
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

#### Full experiment pipeline

To run all the scripts sequentially, as we did in our experiments, you can use the [provided bash script](src/scripts/run_paper_matrix.sh):

1. Navigate to the project [`src/`](src/) directory.

2. Execute the script:

```bash
# Make the script executable
chmod +x scripts/run_paper_matrix.sh
# Run the script
./scripts/run_paper_matrix.sh phaseA # For 24h experiments
./scripts/run_paper_matrix.sh phaseB # For 168h experiments
```

#### Run individual agents

To run individual scripts, navigate to the [`src/`](src/) directory and execute the desired agent script. For example:

```bash
python -m ml.bandit_contextual_rh --config configs/network_porto20_s600.yaml --hours 24 --episodes 10 --scenario baseline --out out/test.json
```
