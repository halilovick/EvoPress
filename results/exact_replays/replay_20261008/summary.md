# Exact-budget replay summary

Completed jobs: 97 of 97.

## RQ3: interaction contrasts (lower metric is better; I = -(delta_A + delta_B))

| A | B | fill | tier | W2 NLL I | delta_A | delta_B | C4 NLL I | repair genes (max) | repair range W2 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E3s0 | E3s1 | owner | B | -0.0085 | +0.0032 | +0.0053 | -0.0069 | 3.0 | +0.0017 |
| E3s0 | E3s2 | owner | B | -0.0479 | +0.0078 | +0.0401 | -0.0271 | 6.0 | +0.0074 |
| E3s1 | DWs0 | donor | A | -0.0363 | +0.0079 | +0.0284 | -0.0191 | 1.0 | +0.0025 |
| E3s1 | DWs0 | owner | A | -0.0138 | +0.0079 | +0.0059 | +0.0074 | 0.0 | +0.0000 |
| E3s1 | E3s2 | owner | B | -0.0241 | +0.0079 | +0.0162 | -0.0112 | 3.0 | +0.0101 |
| J12s0 | J12s1 | donor | A | -0.0185 | +0.0513 | -0.0328 | -0.0431 | 4.0 | +0.0534 |
| J12s0 | J12s1 | owner | A | +0.0059 | +0.0447 | -0.0505 | -0.0040 | 0.0 | +0.0000 |
| J12s0 | J12s2 | owner | B | -0.0194 | +0.0131 | +0.0063 | -0.0097 | 4.0 | +0.0041 |
| J12s1 | J12s2 | owner | B | -0.0301 | -0.0407 | +0.0707 | -0.0165 | 2.0 | +0.0012 |

## RQ1 baselines and references

| Job | Precision | W2 PPL | C4 PPL | repair genes |
| --- | --- | ---: | ---: | ---: |
| `own_E3s0` | quantized | 8.9453125 | 12.640625 | 0 |
| `own_E3s1` | quantized | 8.828125 | 12.328125 | 0 |
| `own_E3s2` | quantized | 9.109375 | 12.5234375 | 0 |
| `own_J12s0` | quantized | 6.4140625 | 9.6015625 | 0 |
| `own_J12s1` | quantized | 6.89453125 | 9.734375 | 0 |
| `own_J12s2` | quantized | 6.5546875 | 9.6015625 | 0 |
| `own_DWs0` | quantized | 9.234375 | 12.6640625 | 0 |
| `own_uniform3` | quantized | 5.5 | 8.5546875 | 0 |
| `h25_late_layer_nu` | quantized | 455.25 | 342.5 | 0 |
| `h25_late_layer_keep_last_nu` | quantized | 591.5 | 514.0 | 0 |
| `h25_random_s0_nu` | quantized | 28.21875 | 37.59375 | 0 |
| `h25_random_s1_nu` | quantized | 36.59375 | 49.40625 | 0 |
| `h25_random_s2_nu` | quantized | 40192.0 | 5396.0 | 0 |
| `h25_bi_score_nu` | quantized | 5316.0 | 6876.0 | 0 |
| `h12_late_layer_nu` | quantized | 54.8125 | 82.3125 | 0 |
| `h12_late_layer_keep_last_nu` | quantized | 116.5 | 135.625 | 0 |
| `h12_random_s0_nu` | quantized | 15.34375 | 16.6875 | 0 |
| `h12_random_s1_nu` | quantized | 25.234375 | 31.109375 | 0 |
| `h12_random_s2_nu` | quantized | 19.890625 | 30.984375 | 0 |
| `h12_bi_score_nu` | quantized | 8.1015625 | 11.2890625 | 0 |
| `ind_DO25s0_QE2s0_shift` | quantized | 9.6015625 | 14.2734375 | 1 |
| `ind_DO25s0_nu` | quantized | 10.296875 | 14.6953125 | 0 |
| `ind_DO25s1_QE2s1_shift` | quantized | 10.421875 | 15.28125 | 1 |
| `ind_DO25s1_nu` | quantized | 10.6015625 | 15.4921875 | 0 |
| `ind_DO25s2_QE2s2_shift` | quantized | 9.0859375 | 13.25 | 1 |
| `ind_DO25s2_nu` | quantized | 9.359375 | 13.328125 | 0 |
| `att_E3s0_nu` | quantized | 9.125 | 12.7421875 | 0 |
| `att_E3s0_QE2s0_shift` | quantized | 9.0546875 | 12.71875 | 1 |
| `att_E3s1_nu` | quantized | 8.9140625 | 12.3984375 | 0 |
| `att_E3s1_QE2s1_shift` | quantized | 8.828125 | 12.28125 | 1 |
| `att_E3s2_nu` | quantized | 9.1953125 | 12.59375 | 0 |
| `att_E3s2_QE2s2_shift` | quantized | 9.0546875 | 12.59375 | 1 |
| `att_J12s0_nu` | quantized | 6.80078125 | 9.9453125 | 0 |
| `att_J12s0_QE2s0_shift` | quantized | 6.515625 | 9.6953125 | 0 |
| `att_J12s1_nu` | quantized | 6.99609375 | 10.21875 | 0 |
| `att_J12s1_QE2s1_shift` | quantized | 6.6953125 | 9.9453125 | 0 |
| `att_J12s2_nu` | quantized | 7.03515625 | 10.0234375 | 0 |
| `att_J12s2_QE2s2_shift` | quantized | 6.68359375 | 9.7109375 | 0 |
| `fp16_E3s0` | fp16 | 8.6015625 | 12.15625 | 0 |
| `fp16_E3s1` | fp16 | 8.484375 | 11.9921875 | 0 |
| `fp16_E3s2` | fp16 | 8.7578125 | 12.109375 | 0 |
| `fp16_J12s0` | fp16 | 6.0234375 | 9.015625 | 0 |
| `fp16_J12s1` | fp16 | 6.109375 | 9.15625 | 0 |
| `fp16_J12s2` | fp16 | 6.171875 | 9.0703125 | 0 |
| `fp16_DO25s0` | fp16 | 9.15625 | 13.7265625 | 0 |
| `fp16_DO25s1` | fp16 | 9.9609375 | 14.609375 | 0 |
| `fp16_DO25s2` | fp16 | 8.7734375 | 12.84375 | 0 |
| `fp16_h25_late_layer_keep_last` | fp16 | 639.5 | 514.0 | 0 |
| `fp16_h25_bi_score` | fp16 | 6460.0 | 9040.0 | 0 |
| `fp16_h12_late_layer_keep_last` | fp16 | 93.25 | 120.1875 | 0 |
| `fp16_h12_bi_score` | fp16 | 7.17578125 | 10.2578125 | 0 |
