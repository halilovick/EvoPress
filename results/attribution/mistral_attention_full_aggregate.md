# Joint Compression Attribution Aggregate

| Depth source | Quant source | Runs | WikiText2 PPL mean | C4 PPL mean | FineWeb-Edu PPL mean | Compression mean | Avg active bits | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent | independent | 3/3 | 13.15 | 16.1 | 14.34 | 1.547 | 3 | 0 |
| independent | interaction_aware | 3/3 | 14.19 | 17.4 | 15.43 | 1.547 | 3 | 0 |
| independent | standard_joint | 3/3 | 13.2 | 16.26 | 14.38 | 1.547 | 3 | 0 |
| independent | uniform3 | 3/3 | 13.13 | 16.19 | 14.29 | 1.547 | 3 | 0 |
| interaction_aware | independent | 3/3 | 12.08 | 14.9 | 13.02 | 1.547 | 3 | 0 |
| interaction_aware | interaction_aware | 3/3 | 12.42 | 15.25 | 13.45 | 1.547 | 3 | 0 |
| interaction_aware | standard_joint | 3/3 | 12.3 | 15.14 | 13.12 | 1.547 | 3 | 0 |
| interaction_aware | uniform3 | 3/3 | 12.07 | 14.99 | 12.97 | 1.547 | 3 | 0 |
| standard_joint | independent | 3/3 | 12.7 | 15.63 | 13.78 | 1.547 | 3 | 0 |
| standard_joint | interaction_aware | 3/3 | 13.72 | 17.15 | 15.02 | 1.547 | 3 | 0 |
| standard_joint | standard_joint | 3/3 | 12.67 | 15.59 | 13.78 | 1.547 | 3 | 0 |
| standard_joint | uniform3 | 3/3 | 12.61 | 15.6 | 13.7 | 1.547 | 3 | 0 |

The aggregate includes only rows with `status=completed` in metric means.
