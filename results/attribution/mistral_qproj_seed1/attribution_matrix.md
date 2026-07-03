# Joint Compression Attribution Matrix

| Depth source | Quant source | WikiText2 PPL | C4 PPL | FineWeb-Edu PPL | Compression | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| independent | independent | 12.546875 | 15.734375 | 14.046875 | 1.400303557622533 | completed |
| independent | standard_joint | 12.5234375 | 15.703125 | 14.046875 | 1.400303557622533 | completed |
| independent | interaction_aware | 12.71875 | 15.953125 | 14.1328125 | 1.400303557622533 | completed |
| independent | uniform3 | 12.5234375 | 15.671875 | 14.046875 | 1.400303557622533 | completed |
| standard_joint | independent | 10.859375 | 14.1328125 | 12.203125 | 1.400303557622533 | completed |
| standard_joint | standard_joint | 10.921875 | 14.1875 | 12.2578125 | 1.400303557622533 | completed |
| standard_joint | interaction_aware | 11.1171875 | 14.328125 | 12.421875 | 1.400303557622533 | completed |
| standard_joint | uniform3 | 10.859375 | 14.1328125 | 12.203125 | 1.400303557622533 | completed |
| interaction_aware | independent | 11.03125 | 13.78125 | 11.8515625 | 1.400303557622533 | completed |
| interaction_aware | standard_joint | 11.0703125 | 13.828125 | 11.8515625 | 1.400303557622533 | completed |
| interaction_aware | interaction_aware | 11.1171875 | 13.9375 | 12.015625 | 1.400303557622533 | completed |
| interaction_aware | uniform3 | 11.03125 | 13.8046875 | 11.8515625 | 1.400303557622533 | completed |

Lower perplexity is better. The matrix is post-hoc: it tests replayed
recombinations and does not prove causal mechanisms by itself.
