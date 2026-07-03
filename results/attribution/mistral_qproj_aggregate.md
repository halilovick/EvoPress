# Joint Compression Attribution Aggregate

| Depth source | Quant source | Runs | WikiText2 PPL mean | C4 PPL mean | FineWeb-Edu PPL mean | Compression mean | Avg active bits | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent | independent | 3/3 | 11.96 | 14.89 | 13.15 | 1.4 | 3 | 0 |
| independent | interaction_aware | 3/3 | 12.13 | 15.09 | 13.36 | 1.4 | 3 | 0 |
| independent | standard_joint | 3/3 | 11.96 | 14.94 | 13.15 | 1.4 | 3 | 0 |
| independent | uniform3 | 3/3 | 11.96 | 14.88 | 13.17 | 1.4 | 3 | 0 |
| interaction_aware | independent | 3/3 | 10.95 | 13.94 | 12.05 | 1.4 | 3 | 0 |
| interaction_aware | interaction_aware | 3/3 | 11.09 | 14.13 | 12.22 | 1.4 | 3 | 0 |
| interaction_aware | standard_joint | 3/3 | 10.98 | 13.97 | 12.08 | 1.4 | 3 | 0 |
| interaction_aware | uniform3 | 3/3 | 10.95 | 13.97 | 12.06 | 1.4 | 3 | 0 |
| standard_joint | independent | 3/3 | 11.21 | 14.36 | 12.43 | 1.4 | 3 | 0 |
| standard_joint | interaction_aware | 3/3 | 11.49 | 14.61 | 12.74 | 1.4 | 3 | 0 |
| standard_joint | standard_joint | 3/3 | 11.24 | 14.39 | 12.46 | 1.4 | 3 | 0 |
| standard_joint | uniform3 | 3/3 | 11.21 | 14.38 | 12.44 | 1.4 | 3 | 0 |

The aggregate includes only rows with `status=completed` in metric means.
