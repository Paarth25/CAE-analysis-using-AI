# Suzuki CAE AI Platform – Job No. 10

**AI-Driven CAE Surrogate Modelling and Experimental Validation for Automotive Structural Components**

---

## Overview

This project demonstrates an end-to-end engineering AI/ML workflow for structural product development:

1. **Analytical FEA baseline** – Euler-Bernoulli beam theory for verification
2. **Automated CAE simulation** – Gmsh meshing + CalculiX FEA (requires external installation)
3. **LHS design of experiments** – 600-sample dataset via Latin Hypercube Sampling
4. **Multiple surrogate ML models** – Linear, Polynomial, Random Forest, XGBoost, MLP
5. **Uncertainty quantification** – Bootstrap ensemble prediction intervals
6. **Experimental validation** – Simulation-to-experiment calibration workflow
7. **Active learning** – Uncertainty-guided intelligent simulation selection
8. **Design optimisation** – Mass minimisation with structural constraints
9. **Interactive dashboard** – Streamlit web application

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/suzuki-cae-ai-project.git
cd suzuki-cae-ai-project
pip install -e ".[dev]"

# 2. Generate dataset
python scripts/generate_dataset.py --n-samples 600

# 3. Train all models (three targets)
python scripts/train_models.py --target tip_deflection_mm
python scripts/train_models.py --target max_von_mises_MPa
python scripts/train_models.py --target strain_energy_J

# 4. Run optimisation
python scripts/optimize_design.py --force 50 --stress 200 --delta 5

# 5. Active learning demo
python scripts/run_active_learning.py --n-iter 5

# 6. Launch dashboard
streamlit run app/dashboard.py

# 7. Run all tests
pytest tests/ -v
```

## Repository Structure

```
suzuki-cae-ai-project/
├── src/suzuki_cae/
│   ├── engineering/      # Euler-Bernoulli beam theory, materials
│   ├── cae/              # Gmsh geometry, CalculiX solver, .frd parser
│   ├── data/             # LHS DoE, feasibility filter, preprocessing
│   ├── models/           # Surrogate ML models (6 algorithms)
│   ├── uncertainty/      # Bootstrap ensemble UQ + calibration
│   ├── optimization/     # Design optimizer, active learning
│   ├── analysis/         # EDA, correlation, sensitivity
│   └── utils/            # Units, paths, logging, reproducibility
├── app/
│   ├── dashboard.py      # Home page
│   └── pages/            # 6 Streamlit pages
├── scripts/              # CLI entry points
├── tests/                # 63 automated tests
├── configs/              # YAML configuration
├── docs/                 # Technical documentation
└── data/                 # Raw and processed datasets
```

## Engineering Problem

Rectangular cantilever beam under end load. Design space:

| Parameter | Range |
|-----------|-------|
| Length L | 80–160 mm |
| Width b | 15–40 mm |
| Thickness t | 2–6 mm |
| Young's modulus E | 65–210 GPa |
| Poisson's ratio ν | 0.27–0.35 |
| Applied force F | 5–100 N |

Analytical baselines:
- I = bt³/12
- δ_tip = FL³/(3EI)
- σ_max = 6FL/(bt²)

## Test Results

```
63 passed in 22.95s
```

## Model Performance (tip deflection, test set)

| Model | R² | RMSE (mm) |
|-------|-----|-----------|
| Linear Regression | 0.9995 | 0.062 |
| Polynomial (d=2) | 0.9995 | 0.066 |
| Random Forest | 0.9955 | 0.191 |
| XGBoost | 0.9876 | 0.317 |
| MLP | 0.9929 | 0.239 |
| **Analytical baseline** | **0.9996** | **0.055** |

## Optimisation Result (F=50 N, σ_allow=200 MPa, δ_allow=5 mm)

- Baseline: L=120 mm, b=25 mm, t=4 mm → mass = 32.4 g
- Optimal:  L=80 mm,  b=15 mm, t=2.83 mm → mass = 9.2 g
- **Mass saving: 71.7%** with constraints satisfied

## CAE Integration (requires external installation)

```bash
# Install CalculiX
sudo apt-get install calculix-ccx      # Ubuntu/Debian
brew install calculix                   # macOS

# Install Gmsh
pip install gmsh

# Then run single simulation
python scripts/run_single_simulation.py \
    --length 100 --width 20 --thickness 4 \
    --modulus 70 --poisson 0.33 --force 50
```

## Data Provenance

- `data/raw/cae/beam_dataset_analytical.csv` – generated from analytical solver + LHS
- `data/raw/experimental/synthetic_experimental.csv` – **SYNTHETIC** (not physical measurements)
- Real experimental data: replace with physical dial-gauge measurements on aluminium flat bars

## Limitations

1. Analytical solver used in place of CalculiX where solver is not installed
2. Experimental data is synthetic — not from physical measurement
3. Linear elastic model only (no plasticity, no buckling)
4. Single load case (concentrated end load)
5. Active learning improvement is modest at small dataset sizes

## License

MIT License — see LICENSE file.