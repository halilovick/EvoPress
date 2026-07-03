# Joint Compression Attribution Matrix

| Depth source | Quant source | WikiText2 PPL | C4 PPL | FineWeb-Edu PPL | Compression | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| independent | independent | 11.3515625 | 14.1328125 | 12.421875 | 1.400303557622533 | completed |
| independent | standard_joint | 11.3515625 | 14.1875 | 12.421875 | 1.400303557622533 | completed |
| independent | interaction_aware | 11.46875 | 14.296875 | 12.6953125 | 1.400303557622533 | completed |
| independent | uniform3 | 11.3515625 | 14.1015625 | 12.4453125 | 1.400303557622533 | completed |
| standard_joint | independent | 11.2890625 | 14.5546875 | 12.6953125 | 1.400303557622533 | completed |
| standard_joint | standard_joint | 11.3359375 | 14.5546875 | 12.6953125 | 1.400303557622533 | completed |
| standard_joint | interaction_aware | 11.671875 | 14.8671875 | 13.0703125 | 1.400303557622533 | completed |
| standard_joint | uniform3 | 11.265625 | 14.578125 | 12.6953125 | 1.400303557622533 | completed |
| interaction_aware | independent | 11.046875 | 13.9921875 | 12.109375 | 1.400303557622533 | completed |
| interaction_aware | standard_joint | 11.0703125 | 13.96875 | 12.1796875 | 1.400303557622533 | completed |
| interaction_aware | interaction_aware | 11.2421875 | 14.2734375 | 12.3046875 | 1.400303557622533 | completed |
| interaction_aware | uniform3 | 11.046875 | 14.0234375 | 12.1328125 | 1.400303557622533 | completed |

Lower perplexity is better. The matrix is post-hoc: it tests replayed
recombinations and does not prove causal mechanisms by itself.
