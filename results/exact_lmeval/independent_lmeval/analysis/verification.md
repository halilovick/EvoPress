# LM-eval verification

- [x] all 6 jobs completed: 6/6
- [x] no sample limit: none
- [x] identical settings for all jobs: 1 distinct
- [x] settings identical to the 11-model LM-eval: identical
- [x] full task sets {'arc_easy': 2376, 'piqa': 1838, 'winogrande': 1267}: all full
- [x] realized cost = 26,982,023,168 bits: all quantized jobs
- [x] bit-widths identical to the replay batch: compared 6: ['ind_DO12s0_QE2s0_shift', 'ind_DO12s1_QE2s1_shift', 'ind_DO12s2_QE2s2_shift', 'ind_DO25s0_QE2s0_shift', 'ind_DO25s1_QE2s1_shift', 'ind_DO25s2_QE2s2_shift']; mismatched: []
- [x] removed sublayers (attn, MLP) as expected: ok
