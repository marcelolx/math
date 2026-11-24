# math

> **Work-in-progress open research notebook.** Expect rough scripts, partial results, and documented dead ends. Everything here is shared in the spirit of *open notebook science*—reproducible negative results are just as valuable as breakthroughs.

## What This Repo Is

An open, general-purpose research notebook for tackling narrow, verifiable problems across mathematics, physics, and engineering.

Instead of broad speculation, each investigation follows a tight local **write → run → verify** loop:

1. **Shrink the target** to a specific, testable mathematical or physical sub-problem.
2. **State the math & assumptions** explicitly (including intentional simplifications like 1D/2D reductions or toy models).
3. **Verify locally** with runnable simulation scripts (Python) or formal proofs (Lean) and inspect raw outputs.
4. **Log the outcome**—whether it works, partially works, or blows up—so anyone can clone the repo, run the code, and pick up where we left off.

## Repository Layout

Each research track lives in its own self-contained folder:

```text
math/
├── README.md
├── LICENSE
├── .gitignore
├── templates/
│   └── progress-block.md   # Standardized session log template
└── <topic>/
    ├── entries/            # Numbered Progress Blocks (000-baseline.md, ...)
    ├── sim/                # Runnable Python scripts, notebooks, or Lean files
    └── results/            # Lightweight plots, metrics, and summary outputs
```

> **Note for Python scripts:** Avoid naming any file `math.py` or package directory `math/` inside the repo so Python's standard library `math` module is never shadowed.

## Topics

| Topic | Focus | Status |
| :--- | :--- | :--- |
| [`rocket-combustion/`](./rocket-combustion/) | Thermoacoustic combustion instability, passive cavities, baffles & nonlinear triggering (Entries `000`–`006` + [`SYNTHESIS.md`](./rocket-combustion/SYNTHESIS.md)) | Active |
| [`plasma-nozzles/`](./plasma-nozzles/) | Magnetic nozzle acceleration, 2D resistive-MHD shear stability & compliant-tube detachment (Entries `000`–`002` + [`SYNTHESIS.md`](./plasma-nozzles/SYNTHESIS.md)) | Active |
| [`grid-stability/`](./grid-stability/) | Renewable AC power-grid synchronization, Braess's paradox & Grid-Forming (GFM) inverter stabilization (Entries `000`–`001` + [`SYNTHESIS.md`](./grid-stability/SYNTHESIS.md)) | Active |

## Running Simulations Locally

Python dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python rocket-combustion/sim/baseline_1d_crocco.py
```

## License

- **Code & simulations:** [MIT License](./LICENSE)
- **Written notes, derivations & figures:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
