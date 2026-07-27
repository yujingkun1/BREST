# Prediction-calibrated bulk completion

Let `O` be observed panel genes and `H` be off-panel or held-out genes. BREST
estimates a covariance matrix from a matched bulk RNA-seq cohort and applies:

```python
from brest.downstream import bulk_covariance, impute_blup

covariance = bulk_covariance(bulk_expression_z)
heldout_prediction = impute_blup(
    observed_expression_z,
    covariance,
    observed_indices,
    heldout_indices,
    lam=1.0,
)
```

Use `lam=10.0` for the measured-panel oracle and `lam=1.0` for
BREST-predicted observed genes. The latter is the reported `Bulk-pred` route.

To estimate the prediction noise from training data:

```python
from brest.downstream import estimate_prediction_ridge

ridge = estimate_prediction_ridge(
    training_prediction_z,
    training_target_z,
    observed_indices,
)
```

The estimator reads observed training genes only. Development estimates were
close to one, so the reported Visium and Xenium experiments use a fixed
`lam=1.0` without per-test-dataset tuning.

The fixed mask-100 protocol samples 100 panel genes with
`numpy.random.RandomState(0)`. Direct prediction, measured-panel completion,
and prediction-based completion are evaluated on exactly those genes with
Overall PCC and mean Gene PCC.
