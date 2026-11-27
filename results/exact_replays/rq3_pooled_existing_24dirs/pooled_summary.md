# RQ3 pooled analysis: shared-gene exchange minus deficit-matched null

excess = mean(real) - mean(null), log WikiText-2 PPL (0.01 = about 1% perplexity) and calibration KL.
Equivalence is judged on the mean ABSOLUTE excess: a signed mean lets opposite directions cancel.
CI: bootstrap over pairs (both directions resampled together); model-level CI resamples final models.

**noIA / wikitext2_ppl**: 14 directions, 7 pairs; mean excess +0.0015; 90% CI (pairs) [-0.0017, +0.0048]; 90% CI (models) [-0.0034, +0.0076]; mean |excess| 0.0114 (95% upper bound 0.0260); largest |excess| 0.0672; directions within margin: ±0.01: 12/14, ±0.02: 12/14; mean |excess| upper bound below margin: ±0.01: no, ±0.02: no

**noIA / calibration_kl**: 14 directions, 7 pairs; mean excess +0.0018; 90% CI (pairs) [-0.0006, +0.0051]; 90% CI (models) [-0.0013, +0.0080]; mean |excess| 0.0044 (95% upper bound 0.0068); largest |excess| 0.0159; directions within margin: ±0.01: 13/14, ±0.02: 14/14; mean |excess| upper bound below margin: ±0.01: yes, ±0.02: yes

**IA / wikitext2_ppl**: 10 directions, 5 pairs; mean excess -0.0009; 90% CI (pairs) [-0.0099, +0.0082]; 90% CI (models) [-0.0118, +0.0167]; mean |excess| 0.0844 (95% upper bound 0.1191); largest |excess| 0.1838; directions within margin: ±0.01: 0/10, ±0.02: 1/10; mean |excess| upper bound below margin: ±0.01: no, ±0.02: no

**IA / calibration_kl**: 10 directions, 5 pairs; mean excess +0.0007; 90% CI (pairs) [-0.0037, +0.0049]; 90% CI (models) [-0.0048, +0.0055]; mean |excess| 0.0593 (95% upper bound 0.0713); largest |excess| 0.0817; directions within margin: ±0.01: 0/10, ±0.02: 0/10; mean |excess| upper bound below margin: ±0.01: no, ±0.02: no

**all / wikitext2_ppl**: 24 directions, 12 pairs; mean excess +0.0005; 90% CI (pairs) [-0.0038, +0.0048]; 90% CI (models) [-0.0066, +0.0078]; mean |excess| 0.0418 (95% upper bound 0.0666); largest |excess| 0.1838; directions within margin: ±0.01: 12/24, ±0.02: 13/24; mean |excess| upper bound below margin: ±0.01: no, ±0.02: no

**all / calibration_kl**: 24 directions, 12 pairs; mean excess +0.0014; 90% CI (pairs) [-0.0011, +0.0039]; 90% CI (models) [-0.0023, +0.0053]; mean |excess| 0.0273 (95% upper bound 0.0420); largest |excess| 0.0817; directions within margin: ±0.01: 13/24, ±0.02: 14/24; mean |excess| upper bound below margin: ±0.01: no, ±0.02: no

## Direction pairs: sum of the two directions (W2 log PPL)

| Pair | group | sum | |excess| (each) |
| --- | --- | ---: | --- |
| DWs0xIAs0 | IA | -0.0346 | 0.0622 / 0.0969 |
| E3s0xIAs2 | IA | +0.0334 | 0.1504 / 0.1838 |
| E3s2xIAs0 | IA | -0.0202 | 0.0711 / 0.0913 |
| IAs0xIAs1 | IA | -0.0049 | 0.0156 / 0.0205 |
| IAs0xLXs0 | IA | +0.0177 | 0.0675 / 0.0851 |
| E3s0xE3s1 | noIA | -0.0052 | 0.0012 / 0.0041 |
| E3s0xE3s2 | noIA | -0.0034 | 0.0000 / 0.0035 |
| E3s0xLXs0 | noIA | +0.0092 | 0.0008 / 0.0084 |
| E3s1xE3s2 | noIA | +0.0113 | 0.0042 / 0.0071 |
| E3s1xP4s1 | noIA | -0.0067 | 0.0023 / 0.0091 |
| J12s0xJ12s2 | noIA | -0.0071 | 0.0010 / 0.0061 |
| J12s1xJ12s2 | noIA | +0.0225 | 0.0446 / 0.0672 |
