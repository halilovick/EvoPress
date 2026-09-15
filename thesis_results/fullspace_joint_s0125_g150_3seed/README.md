# Full-space 12.5% joint compression — G150, 3 seeds

Model: Mistral-7B-v0.3
Search: standard joint, G150 / O128
Structural sparsity: 12.5%
Dropped modules: 4 attention + 4 MLP
Exact storage target: 26,982,023,168 bits
Target size: 3.1411209106 GiB

## Primary three-seed result

WikiText2 PPL:
6.62109375 ± 0.20169267

C4 PPL:
9.64583333 ± 0.06260841

Train PPL:
7.86067708 ± 0.06872731

Final calibration KL:
0.20841471 ± 0.00680121

Dispersion convention used for primary reporting:
population standard deviation.

All three runs satisfy the exact storage budget with zero-bit difference.

Per-seed final quant bit averages reported by the logs:
[3.4405, 3.4657, 3.485]

Mean reported final quant bit average:
3.4637333333333333

See per_seed.csv and summary.json for exact values and provenance.
