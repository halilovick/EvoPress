# RQ3 repair-cost control: real shared-gene exchange vs. deficit-matched random fill

delta = mean(crossed or null) - own, in log PPL / KL (lower is better, so a positive delta is a loss).
excess = delta_real - delta_null. Near 0: the real exchange's cost is explained by perturbing about
this many genes by about this much, regardless of source (confirms the upper-bound reading of
replay-findings-2026-10-09.md, point 9). Clearly positive: the real exchange costs more than
equally-sized random noise -- evidence of a genuine mask x quantization interaction.

| Mask | vs | genes (real/null) | W2 NLL δreal | δnull | excess | KL δreal | δnull | excess |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E3s0 | LXs0 | 1.0/1.0 | +0.0104 | +0.0020 | +0.0084 | +0.0103 | +0.0007 | +0.0095 |
| LXs0 | E3s0 | 1.0/1.0 | +0.0041 | +0.0033 | +0.0008 | +0.0170 | +0.0011 | +0.0159 |
| IAs0 | LXs0 | 2.0/2.0 | -0.0655 | +0.0019 | -0.0675 | -0.0521 | +0.0047 | -0.0568 |
| LXs0 | IAs0 | 2.0/2.0 | +0.0903 | +0.0052 | +0.0851 | +0.0799 | +0.0040 | +0.0759 |
| E3s2 | IAs0 | 2.0/2.0 | +0.0802 | +0.0091 | +0.0711 | +0.0708 | +0.0050 | +0.0658 |
| IAs0 | E3s2 | 2.0/2.0 | -0.0650 | +0.0263 | -0.0913 | -0.0470 | +0.0130 | -0.0601 |
| IAs0 | IAs1 | 3.3/3.3 | -0.0099 | +0.0106 | -0.0205 | -0.0153 | +0.0070 | -0.0223 |
| IAs1 | IAs0 | 3.0/3.0 | +0.0259 | +0.0103 | +0.0156 | +0.0257 | +0.0041 | +0.0216 |
| E3s1 | P4s1 | 3.7/3.7 | +0.0120 | +0.0097 | +0.0023 | +0.0136 | +0.0072 | +0.0064 |
| P4s1 | E3s1 | 3.7/3.7 | +0.0044 | +0.0135 | -0.0091 | -0.0003 | +0.0074 | -0.0077 |
| IAs0 | DWs0 | 4.0/4.0 | -0.0667 | +0.0301 | -0.0969 | -0.0507 | +0.0234 | -0.0741 |
| DWs0 | IAs0 | 5.7/5.7 | +0.0985 | +0.0363 | +0.0622 | +0.0744 | +0.0193 | +0.0551 |
| E3s0 | IAs2 | 7.0/7.0 | +0.1925 | +0.0087 | +0.1838 | +0.0912 | +0.0095 | +0.0817 |
| IAs2 | E3s0 | 5.3/5.3 | -0.1167 | +0.0337 | -0.1504 | -0.0697 | +0.0096 | -0.0793 |

Mean W2 NLL excess over 14 directions: -0.0004 (population SD 0.0826).
