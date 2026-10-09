# LM-eval verification

- [x] all 11 jobs completed: 11/11
- [x] no sample limit: none
- [x] identical settings for all jobs: 1 distinct
- [x] full task sets {'arc_easy': 2376, 'piqa': 1838, 'winogrande': 1267}: all full
- [x] realized cost = 26,982,023,168 bits: all quantized jobs
- [x] bit-widths identical to the replay batch: compared 7: ['own_E3s0', 'own_E3s1', 'own_E3s2', 'own_J12s0', 'own_J12s1', 'own_J12s2', 'own_uniform3']; mismatched: []
- [x] removed sublayers (attn, MLP) as expected: ok
- [x] dense vs screening dense LM-eval (|diff| <= 0.01): {'arc_easy': 0.0008, 'piqa': 0.0011, 'winogrande': 0.0008}
