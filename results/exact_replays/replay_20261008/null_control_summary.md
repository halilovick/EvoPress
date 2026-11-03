# RQ3 repair-cost control: real shared-gene exchange vs. deficit-matched random fill

delta = mean(crossed or null) - own, in log PPL / KL (lower is better, so a positive delta is a loss).
excess = delta_real - delta_null. Near 0: the real exchange's cost is explained by perturbing about
this many genes by about this much, regardless of source (confirms the upper-bound reading of
replay-findings-2026-10-09.md, point 9). Clearly positive: the real exchange costs more than
equally-sized random noise -- evidence of a genuine mask x quantization interaction.

| Mask | vs | genes (real/null) | W2 NLL δreal | δnull | excess | KL δreal | δnull | excess |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E3s0 | E3s1 | 3.0/3.0 | +0.0032 | +0.0073 | -0.0041 | +0.0011 | +0.0033 | -0.0022 |
| E3s1 | E3s0 | 3.0/3.0 | +0.0053 | +0.0065 | -0.0012 | +0.0046 | +0.0023 | +0.0024 |
| E3s0 | E3s2 | 6.0/6.0 | +0.0078 | +0.0113 | -0.0035 | +0.0056 | +0.0099 | -0.0043 |
| E3s2 | E3s0 | 5.0/5.0 | +0.0401 | +0.0400 | +0.0000 | +0.0155 | +0.0142 | +0.0013 |
| E3s1 | E3s2 | 3.0/3.0 | +0.0079 | +0.0009 | +0.0071 | +0.0054 | +0.0023 | +0.0031 |
| E3s2 | E3s1 | 3.0/3.0 | +0.0162 | +0.0119 | +0.0042 | +0.0038 | +0.0045 | -0.0007 |
| J12s0 | J12s2 | 4.0/4.0 | +0.0131 | +0.0141 | -0.0010 | +0.0077 | +0.0072 | +0.0005 |
| J12s2 | J12s0 | 3.0/3.0 | +0.0063 | +0.0124 | -0.0061 | +0.0054 | +0.0085 | -0.0031 |
| J12s1 | J12s2 | 2.0/2.0 | -0.0407 | +0.0040 | -0.0446 | +0.0085 | +0.0065 | +0.0020 |
| J12s2 | J12s1 | 2.0/2.0 | +0.0707 | +0.0036 | +0.0672 | +0.0044 | +0.0023 | +0.0021 |

Mean W2 NLL excess over 10 directions: +0.0018 (population SD 0.0257).
