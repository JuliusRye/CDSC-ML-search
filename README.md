[![Pytest](https://github.com/JuliusRye/CDSC-ML-search/actions/workflows/tests.yml/badge.svg)](https://github.com/JuliusRye/CDSC-ML-search/actions/workflows/tests.yml)

This repository provides tools for training modified convolutional neural networks (mCNNs) as decoders for [Clifford-Deformed Surface Codes](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010347) (CDSCs), a class of quantum error-correcting codes.

This repo is an updated and refined version of the [code I wrote during my master thesis](https://github.com/JuliusRye/QEC).

# Installation

**NOTE**: This code was designed with Python 3.12.3 

**Step 1)** Create a virtual environment: `python3 -m venv .venv` or in VS-Code: "Python: Create Environment..."

**Step 2)** Activate the environment:
Windows: `.venv\Scripts\activate`
MacOS/Linux: `source .venv/bin/activate`

**Step 3)** Install the required packages: `pip install -r requirements.txt`

**Step 4)** If an NVIDIA GPU is available make jax use it: `pip install -U "jax[cuda12]"`

**Step 5)** Verify that everything is working by running: `pytest`

# Usage

Train a **mCNN** on the **CDSCs** by running:

`python experiments/train_mCNN.py <save_decoder_as> <deformation_name> <code_distance> <training_config> <training_batches> [random_seed]`

| Argument             | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `<save_decoder_as>`  | Folder name for storing training results (under `results/`). |
| `<deformation_name>` | Which deformation to use (see list below).                   |
| `<code_distance>`    | Distance `d` of the CDSC.                                    |
| `<training_config>`  | Config file from `experiments/training_configs`.             |
| `<training_batches>` | Number of training rounds.                                   |
| `[random_seed]`      | Optional seed for reproducibility (default `0`).             |

#### Supported options for <deformation_name>:
- "Generalized": Creates a decoder that has been trained uniformly on all **CDSCs**.
- "Best": Starts like the Generalized decoder, but over time it will narrow its focus to the **CDSCs** it believes are the best in terms of logical error rate.
- "CSS": Train only on the undeformed rotated surface code.
- "XZZX": Train only on the [XZZX CDSC](https://www.nature.com/articles/s41467-021-22274-1).
- "XY": Train only on the [XY CDSC](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.041031).
- "C1": Train only on the [C1 CDSC](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010347).
- A string of $d^2$ integers denoting the index of the Clifford deformation to be used on each data qubit.

#### Example usage:

Create a **mCNN** decoder for the distance-3 C1 **CDSCs** for 1M training steps:

`python experiments/train_mCNN.py C1_d3 C1 3 default 1_000_000`

Create a **mCNN** decoder for the distance-5 surface code that can decode any **CDSCs** for 100k training steps (in practise more is needed):

`python experiments/train_mCNN.py gen_d5 Generalized 5 default 100_000`

Create a **mCNN** decoder for the distance-3 surface code that can decode the code with deformations [2,5,1] on the bottom row, [4,2,3] on the middle row, and [2,3,0] on the top row:

`python experiments/train_mCNN.py rand_d3 251423230 3 default 100_000`
