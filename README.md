# Autonomous Solar Panel Cleaning Module

Source code and configuration for solar-panel image detection, dataset preparation, model training, evaluation, and demo inference.

## Repository contents

This repository intentionally excludes datasets, trained model weights, virtual environments, training runs, logs, and generated demo output. Obtain or train model weights separately before running inference or evaluation.

The scripts currently contain local Windows paths. Update those paths to match your machine and dataset before using the related data-preparation or training scripts.

## Setup

Use Python 3.10 or newer, then install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Demo

Run the interactive image/video demo with a local model file:

```powershell
python Execution.py --model path\to\best.pt
```

For a webcam demo:

```powershell
python Execution.py --model path\to\best.pt --camera
```

The demo saves annotated output under `demo_output/`, which is excluded from version control.
