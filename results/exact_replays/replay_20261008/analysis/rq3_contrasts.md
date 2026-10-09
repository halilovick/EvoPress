# RQ3: interaction contrasts at the exact budget

I = (J_AA + J_BB) - (J_AB + J_BA) = -(delta_A + delta_B); J = mean NLL (log PPL) or calibration KL; lower is better.
Negative I: each mask does better with its own allocation than additivity predicts.
Repaired crossings use the mean over 3 repair seeds; 'range' is the largest spread over repair seeds.

| Pair | Crossing | Repair (max genes) | W2 NLL I (δA, δB) | C4 NLL I | KL I (δA, δB) | range W2 / KL |
| --- | --- | --- | --- | --- | --- | --- |
| E3s1 × DWs0 | shared-gene exchange | repair-free (0) | -0.0138 (+0.0079, +0.0059) | +0.0074 | -0.0029 (+0.0066, -0.0037) | 0.0000 / 0.0000 |
| J12s0 × J12s1 | shared-gene exchange | repair-free (0) | +0.0059 (+0.0447, -0.0505) | -0.0040 | -0.0056 (+0.0032, +0.0024) | 0.0000 / 0.0000 |
| E3s1 × DWs0 | plain (D_A, Q_B) | plain, production repair (1) | -0.0363 (+0.0079, +0.0284) | -0.0191 | -0.0197 (+0.0081, +0.0116) | 0.0025 / 0.0042 |
| J12s0 × J12s1 | plain (D_A, Q_B) | plain, production repair (4) | -0.0185 (+0.0513, -0.0328) | -0.0431 | -0.0292 (+0.0136, +0.0156) | 0.0534 / 0.0028 |
| E3s0 × E3s1 | shared-gene exchange | exclusive repair (3) | -0.0085 (+0.0032, +0.0053) | -0.0069 | -0.0057 (+0.0011, +0.0046) | 0.0017 / 0.0015 |
| E3s0 × E3s2 | shared-gene exchange | exclusive repair (6) | -0.0479 (+0.0078, +0.0401) | -0.0271 | -0.0212 (+0.0056, +0.0155) | 0.0074 / 0.0063 |
| E3s1 × E3s2 | shared-gene exchange | exclusive repair (3) | -0.0241 (+0.0079, +0.0162) | -0.0112 | -0.0092 (+0.0054, +0.0038) | 0.0101 / 0.0017 |
| J12s0 × J12s2 | shared-gene exchange | exclusive repair (4) | -0.0194 (+0.0131, +0.0063) | -0.0097 | -0.0131 (+0.0077, +0.0054) | 0.0041 / 0.0029 |
| J12s1 × J12s2 | shared-gene exchange | exclusive repair (2) | -0.0301 (-0.0407, +0.0707) | -0.0165 | -0.0129 (+0.0085, +0.0044) | 0.0012 / 0.0028 |
