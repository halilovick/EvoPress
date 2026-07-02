# Joint Compression Attribution Matrix

| Depth source | Quant source | WikiText2 PPL | C4 PPL | FineWeb-Edu PPL | Compression | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| independent | independent | 11.96875 | 14.8125 | 12.96875 | 1.400303557622533 | completed |
| independent | standard_joint | 12.015625 | 14.9296875 | 12.9921875 | 1.400303557622533 | completed |
| independent | interaction_aware | 12.203125 | 15.015625 | 13.25 | 1.400303557622533 | completed |
| independent | uniform3 | 11.9921875 | 14.8671875 | 13.015625 | 1.400303557622533 | completed |
| standard_joint | independent | 11.46875 | 14.3828125 | 12.3984375 | 1.400303557622533 | completed |
| standard_joint | standard_joint | 11.46875 | 14.4375 | 12.421875 | 1.400303557622533 | completed |
| standard_joint | interaction_aware | 11.6953125 | 14.640625 | 12.7421875 | 1.400303557622533 | completed |
| standard_joint | uniform3 | 11.4921875 | 14.4140625 | 12.421875 | 1.400303557622533 | completed |
| interaction_aware | independent | 10.7734375 | 14.046875 | 12.203125 | 1.400303557622533 | completed |
| interaction_aware | standard_joint | 10.796875 | 14.1015625 | 12.203125 | 1.400303557622533 | completed |
| interaction_aware | interaction_aware | 10.8984375 | 14.1875 | 12.3515625 | 1.400303557622533 | completed |
| interaction_aware | uniform3 | 10.7734375 | 14.078125 | 12.203125 | 1.400303557622533 | completed |

Lower perplexity is better. The matrix is post-hoc: it tests replayed
recombinations and does not prove causal mechanisms by itself.
