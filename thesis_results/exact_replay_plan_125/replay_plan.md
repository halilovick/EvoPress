# Exact-budget replay plan

Jobs: 106

## Tier 0 (8 jobs)

| Job | RQ | Purpose | Repair |
| --- | --- | --- | --- |
| `own_E3s0` | control | re-evaluate completed final candidate | none |
| `own_E3s1` | control | re-evaluate completed final candidate | none |
| `own_E3s2` | control | re-evaluate completed final candidate | none |
| `own_J12s0` | control | re-evaluate completed final candidate | none |
| `own_J12s1` | control | re-evaluate completed final candidate | none |
| `own_J12s2` | control | re-evaluate completed final candidate | none |
| `own_DWs0` | control | re-evaluate completed final candidate | none |
| `own_uniform3` | control | re-evaluate E1 uniform 3-bit | none |

## Tier A (16 jobs)

| Job | RQ | Purpose | Repair |
| --- | --- | --- | --- |
| `x_owner_exclusive_E3s1_QDWs0_r0` | RQ3 | shared-gene exchange, repair-free | none |
| `x_owner_exclusive_DWs0_QE3s1_r0` | RQ3 | shared-gene exchange, repair-free | none |
| `x_owner_exclusive_J12s0_QJ12s1_r0` | RQ3 | shared-gene exchange, repair-free | none |
| `x_owner_exclusive_J12s1_QJ12s0_r0` | RQ3 | shared-gene exchange, repair-free | none |
| `x_donor_all_E3s1_QDWs0_r0` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_E3s1_QDWs0_r1` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_E3s1_QDWs0_r2` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_DWs0_QE3s1_r0` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_DWs0_QE3s1_r1` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_DWs0_QE3s1_r2` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_J12s0_QJ12s1_r0` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_J12s0_QJ12s1_r1` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_J12s0_QJ12s1_r2` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_J12s1_QJ12s0_r0` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_J12s1_QJ12s0_r1` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |
| `x_donor_all_J12s1_QJ12s0_r2` | RQ3 | plain crossing (D_A, Q_B) of the same pair | all |

## Tier B (30 jobs)

| Job | RQ | Purpose | Repair |
| --- | --- | --- | --- |
| `x_owner_exclusive_E3s0_QE3s1_r0` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s0_QE3s1_r1` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s0_QE3s1_r2` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s1_QE3s0_r0` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s1_QE3s0_r1` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s1_QE3s0_r2` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s0_QE3s2_r0` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s0_QE3s2_r1` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s0_QE3s2_r2` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s2_QE3s0_r0` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s2_QE3s0_r1` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s2_QE3s0_r2` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s1_QE3s2_r0` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s1_QE3s2_r1` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s1_QE3s2_r2` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s2_QE3s1_r0` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s2_QE3s1_r1` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_E3s2_QE3s1_r2` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s0_QJ12s2_r0` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s0_QJ12s2_r1` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s0_QJ12s2_r2` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s2_QJ12s0_r0` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s2_QJ12s0_r1` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s2_QJ12s0_r2` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s1_QJ12s2_r0` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s1_QJ12s2_r1` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s1_QJ12s2_r2` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s2_QJ12s1_r0` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s2_QJ12s1_r1` | RQ3 | shared-gene exchange, minimal repair | exclusive |
| `x_owner_exclusive_J12s2_QJ12s1_r2` | RQ3 | shared-gene exchange, minimal repair | exclusive |

## Tier C (36 jobs)

| Job | RQ | Purpose | Repair |
| --- | --- | --- | --- |
| `h25_late_layer_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h25_late_layer_keep_last_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h25_random_s0_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h25_random_s1_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h25_random_s2_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h25_bi_score_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h12_late_layer_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h12_late_layer_keep_last_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h12_random_s0_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h12_random_s1_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h12_random_s2_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `h12_bi_score_nu` | RQ1 | heuristic mask + near-uniform precision | none |
| `ind_DO25s0_QE2s0_shift` | RQ1 | independent composition: depth-only mask, quantization-only profile shifted to budget | all |
| `ind_DO25s0_nu` | RQ1 | depth-only mask + near-uniform precision | none |
| `ind_DO25s1_QE2s1_shift` | RQ1 | independent composition: depth-only mask, quantization-only profile shifted to budget | all |
| `ind_DO25s1_nu` | RQ1 | depth-only mask + near-uniform precision | none |
| `ind_DO25s2_QE2s2_shift` | RQ1 | independent composition: depth-only mask, quantization-only profile shifted to budget | all |
| `ind_DO25s2_nu` | RQ1 | depth-only mask + near-uniform precision | none |
| `ind_DO12s0_QE2s0_shift` | RQ1 | independent composition: depth-only mask, quantization-only profile shifted to budget | all |
| `ind_DO12s0_nu` | RQ1 | depth-only mask + near-uniform precision | none |
| `ind_DO12s1_QE2s1_shift` | RQ1 | independent composition: depth-only mask, quantization-only profile shifted to budget | all |
| `ind_DO12s1_nu` | RQ1 | depth-only mask + near-uniform precision | none |
| `ind_DO12s2_QE2s2_shift` | RQ1 | independent composition: depth-only mask, quantization-only profile shifted to budget | all |
| `ind_DO12s2_nu` | RQ1 | depth-only mask + near-uniform precision | none |
| `att_E3s0_nu` | RQ1/RQ3 | joint mask + near-uniform precision | none |
| `att_E3s0_QE2s0_shift` | RQ1/RQ3 | joint mask + quantization-only profile shifted | all |
| `att_E3s1_nu` | RQ1/RQ3 | joint mask + near-uniform precision | none |
| `att_E3s1_QE2s1_shift` | RQ1/RQ3 | joint mask + quantization-only profile shifted | all |
| `att_E3s2_nu` | RQ1/RQ3 | joint mask + near-uniform precision | none |
| `att_E3s2_QE2s2_shift` | RQ1/RQ3 | joint mask + quantization-only profile shifted | all |
| `att_J12s0_nu` | RQ1/RQ3 | joint mask + near-uniform precision | none |
| `att_J12s0_QE2s0_shift` | RQ1/RQ3 | joint mask + quantization-only profile shifted | all |
| `att_J12s1_nu` | RQ1/RQ3 | joint mask + near-uniform precision | none |
| `att_J12s1_QE2s1_shift` | RQ1/RQ3 | joint mask + quantization-only profile shifted | all |
| `att_J12s2_nu` | RQ1/RQ3 | joint mask + near-uniform precision | none |
| `att_J12s2_QE2s2_shift` | RQ1/RQ3 | joint mask + quantization-only profile shifted | all |

## Tier D (16 jobs)

| Job | RQ | Purpose | Repair |
| --- | --- | --- | --- |
| `fp16_E3s0` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_E3s1` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_E3s2` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_J12s0` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_J12s1` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_J12s2` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_DO25s0` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_DO25s1` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_DO25s2` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_DO12s0` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_DO12s1` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_DO12s2` | RQ1 reference | 16-bit depth-only evaluation of a mask | - |
| `fp16_h25_late_layer_keep_last` | RQ1 reference | 16-bit heuristic depth-only evaluation | - |
| `fp16_h25_bi_score` | RQ1 reference | 16-bit heuristic depth-only evaluation | - |
| `fp16_h12_late_layer_keep_last` | RQ1 reference | 16-bit heuristic depth-only evaluation | - |
| `fp16_h12_bi_score` | RQ1 reference | 16-bit heuristic depth-only evaluation | - |

