[![Pytest](https://github.com/JuliusRye/CDSC-ML-search/actions/workflows/tests.yml/badge.svg)](https://github.com/JuliusRye/CDSC-ML-search/actions/workflows/tests.yml)

This repo is an updated and refined version of the [code I wrote during my master thesis](https://github.com/JuliusRye/QEC).


# Installation

**NOTE**: This code was designed with Python 3.12.3 

**Step 1)** Create a virtual environment:
`python3 -m venv .venv` or in VS-Code: "Python: Create Environment..."

**Step 2)** Activate the environment:
Windows: `.venv\Scripts\activate`
MacOS/Linux: `source .venv/bin/activate`

**Step 3)** Install the required packages:
`pip install -r requirements.txt`

**Step 4)** If an NVIDIA CPU is available make jax use it:
`pip install -U "jax[cuda12]"`

**Step 5)** Verify that everything is working by running:
`pytest`