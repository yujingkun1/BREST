# BREST executable demo

Open [`BREST_demo.ipynb`](BREST_demo.ipynb) for the shortest complete path
through spatial prediction and bulk-prior completion.

## Files

| File | Purpose |
|---|---|
| `BREST_demo.ipynb` | executed end-to-end tutorial with plots |
| `build_demo_data.py` | deterministic synthetic cohort generator |
| `demo_utils.py` | training, prediction, and completion helpers used by the notebook |
| `make_notebook.py` | reproducibly builds the notebook source |
| `validate_demo.py` | checks data checksum, schema, notebook execution, and errors |
| `demo_data/` | 4.28 MiB redistributable input plus manifest |

## Reproduce

From the repository root:

```bash
python demo/build_demo_data.py
python demo/make_notebook.py
jupyter nbconvert --to notebook --execute demo/BREST_demo.ipynb \
  --inplace --ExecutePreprocessor.timeout=600
python demo/validate_demo.py
```

The notebook auto-selects CUDA when available and otherwise runs on CPU.
Numerical values are expected to be deterministic on the same software and
hardware stack, but minor floating-point variation across devices is normal.

Do not cite the toy scores as experimental results. Full input formats and
evaluation protocols are documented under [`docs/`](../docs/).
