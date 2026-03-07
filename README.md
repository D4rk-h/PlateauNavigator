# PlateauNavigator
> **⚠️ Under Active Development** - This project is currently being built.

A research framework for studying VQE behavior and comparing performance between **discrete variable (DV)** and **continuous variable (CV)** quantum computing approaches across different ansatz configurations.

## What is this?

PlateauNavigator is a study tool built to investigate how VQE performs differently depending on whether you're working in the DV or CV quantum computing paradigm. It lets you run experiments across multiple ansatz configurations, collect results, and visualize them — with the goal of understanding things like barren plateau behavior, convergence, and optimization landscape differences between the two paradigms.

It combines:
- A high-performance quantum simulator (Java backend, up to 25 qubits)
- Support for multiple backends (Java simulator, Qiskit, PennyLane, IBM Quantum)
- A Python layer for running experiments, analyzing results, and plotting
- ML-based tools to study and predict barren plateau occurrence

## Architecture
```
┌─────────────────────────────────────┐
│   Python Research Layer             │
│   (Experiments, Analysis, Viz)      │
└─────────────┬───────────────────────┘
              │ REST API
┌─────────────▼───────────────────────┐
│   Java Backend                      │
│   (High-performance simulator)      │
└─────────────────────────────────────┘
```

## Research Goal

Compare and study VQE performance between the DV and CV quantum computing paradigms by:
1. Running VQE experiments across different ansatz configurations
2. Analyzing how barren plateaus manifest differently in each paradigm
3. Visualizing and plotting results to surface meaningful behavioral differences
4. Validating findings on real quantum hardware

---

## License

Apache 2.0 [LICENSE]

---

**Note**: This is a research project in active development. Star ⭐ and watch for updates!
