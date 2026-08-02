# Minimal Component Crossover

EvoPress normally retains one parent, so there is no second persistent candidate
from which to construct crossover offspring. This pilot optionally retains a small
uniformly sampled population while leaving the default single-parent search intact.

## Operator

For parents `A = (depth_A, quant_A)` and `B = (depth_B, quant_B)`, component
crossover deep-copies either `(depth_A, quant_B)` or `(depth_B, quant_A)`. The
depth mask is preserved exactly. With active quantization budgeting, the imported
quantization assignment is repaired for that mask, and changed genes are counted.
A changed result is reported as `component crossover + repair`.

The final selection stage adds every persistent parent for elitism and selects the
configured population size. Intermediate selection-stage counts are unchanged.

## CLI

- `--population_size` (default `1`)
- `--crossover_probability` (default `0.0`)
- `--crossover_type component`

Example:

```bash
python evo_joint_search.py ... \
  --population_size 4 \
  --crossover_probability 0.25 \
  --crossover_type component
```

## Limitations

Only component crossover is implemented. Parent selection is uniform. Sequential
initialization modes reject persistent-population/crossover settings. This pilot
does not add gene-level crossover, adaptive crossover, selection studies, or
diversity algorithms.

## Tests executed

- `python -m unittest tests.test_component_crossover`
- `python -m unittest tests.test_run_joint_search_tiny`
- `python -m unittest tests.test_joint_aware_mutation tests.test_sequential_search tests.test_run_sequential_search`
- `python -m py_compile evo_joint_search.py tests/test_component_crossover.py tests/test_run_joint_search_tiny.py`
- `ruff check evo_joint_search.py tests/test_component_crossover.py tests/test_run_joint_search_tiny.py`
- `bash -n scripts/run_joint_search_tiny.sh`
- `git diff --check`
