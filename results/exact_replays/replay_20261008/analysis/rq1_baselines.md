# RQ1 at the exact budget (T = 26,982,023,168 bits)

Mean ± population SD over the listed candidates; individual values when they span more than a factor of 3.
W2 = WikiText-2 test PPL, C4 = C4 validation PPL (length 8,192); KL = calibration KL to the dense model.

| s | Configuration | n | W2 PPL | C4 PPL | Calib. KL |
| --- | --- | ---: | --- | --- | --- |
| – | Dense 16-bit (E0) | 1 | 4.830 | 7.710 | – |
| 0.0 | Uniform 3-bit (E1), replayed | 1 | 5.500 | 8.555 | 0.0904 |
| 0.0 | Quantization-only search (E2) | 3 | 5.229 ± 0.005 | 8.411 ± 0.007 | 0.0731 ± 0.0002 |
| 0.125 | Joint search (J12), replayed | 3 | 6.621 ± 0.202 | 9.646 ± 0.063 | 0.2084 ± 0.0068 |
| 0.125 | Joint masks + E2 profile shifted | 3 | 6.632 ± 0.082 | 9.784 ± 0.114 | 0.2204 ± 0.0087 |
| 0.125 | Joint masks + near-uniform | 3 | 6.944 ± 0.103 | 10.062 ± 0.115 | 0.2384 ± 0.0089 |
| 0.125 | Block-influence mask + near-uniform | 1 | 8.102 | 11.289 | 0.3462 |
| 0.125 | Random masks + near-uniform | 3 | 20.156 ± 4.042 | 26.260 ± 6.769 | 1.0526 ± 0.2192 |
| 0.125 | Last sublayers + near-uniform | 1 | 54.812 | 82.312 | 2.3691 |
| 0.125 | Last sublayers before final + near-uniform | 1 | 116.5 | 135.6 | 2.7051 |
| 0.25 | Joint search (E3), replayed | 3 | 8.961 ± 0.115 | 12.497 ± 0.129 | 0.4613 ± 0.0045 |
| 0.25 | Joint masks + E2 profile shifted | 3 | 8.979 ± 0.107 | 12.531 ± 0.184 | 0.4640 ± 0.0072 |
| 0.25 | Joint masks + near-uniform | 3 | 9.078 ± 0.120 | 12.578 ± 0.141 | 0.4691 ± 0.0055 |
| 0.25 | Independent: depth-only masks + E2 profile shifted | 3 | 9.703 ± 0.550 | 14.268 ± 0.829 | 0.5711 ± 0.0517 |
| 0.25 | Depth-only masks + near-uniform | 3 | 10.086 ± 0.529 | 14.505 ± 0.894 | 0.5863 ± 0.0518 |
| 0.25 | Block-influence mask + near-uniform | 1 | 5316.0 | 6876.0 | 5.3594 |
| 0.25 | Random masks + near-uniform | 3 | 28.22 / 36.59 / 4.019e+04 | 37.59 / 49.41 / 5396 | 3.0537 ± 2.3049 |
| 0.25 | Last sublayers + near-uniform | 1 | 455.2 | 342.5 | 3.9609 |
| 0.25 | Last sublayers before final + near-uniform | 1 | 591.5 | 514.0 | 4.0625 |

16-bit references (same masks, all active projections in 16-bit; not at the budget):

| s | Configuration | n | W2 PPL | C4 PPL | Calib. KL |
| --- | --- | ---: | --- | --- | --- |
| 0.125 | Joint masks, 16-bit (not at budget) | 3 | 6.102 ± 0.061 | 9.081 ± 0.058 | 0.1619 ± 0.0062 |
| 0.125 | Block-influence mask, 16-bit (not at budget) | 1 | 7.176 | 10.258 | 0.2749 |
| 0.125 | Last sublayers before final, 16-bit (not at budget) | 1 | 93.250 | 120.2 | 2.5723 |
| 0.25 | Joint masks, 16-bit (not at budget) | 3 | 8.615 ± 0.112 | 12.086 ± 0.069 | 0.4338 ± 0.0051 |
| 0.25 | Depth-only masks, 16-bit (not at budget) | 3 | 9.297 ± 0.495 | 13.727 ± 0.721 | 0.5400 ± 0.0509 |
| 0.25 | Block-influence mask, 16-bit (not at budget) | 1 | 6460.0 | 9040.0 | 5.6641 |
| 0.25 | Last sublayers before final, 16-bit (not at budget) | 1 | 639.5 | 514.0 | 4.0781 |
