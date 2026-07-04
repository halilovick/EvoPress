# Joint Compression Attribution Matrix

| Depth source | Quant source | WikiText2 PPL | C4 PPL | FineWeb-Edu PPL | Compression | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| independent | independent | 13.9375 | 17.140625 | 15.1875 | 1.5469698122081734 | completed |
| independent | standard_joint | 13.8828125 | 17.453125 | 15.1328125 | 1.5469698122081734 | completed |
| independent | interaction_aware | 14.4140625 | 18.140625 | 15.921875 | 1.5469698122081734 | completed |
| independent | uniform3 | 14.0234375 | 17.25 | 15.1875 | 1.5469698122081734 | completed |
| standard_joint | independent | 11.8046875 | 14.9296875 | 13.09375 | 1.5469698122081734 | completed |
| standard_joint | standard_joint | 11.7421875 | 14.8125 | 12.96875 | 1.5469698122081734 | completed |
| standard_joint | interaction_aware | 11.921875 | 15.28125 | 13.3828125 | 1.5469698122081734 | completed |
| standard_joint | uniform3 | 11.6953125 | 14.78125 | 12.8203125 | 1.5469698122081734 | completed |
| interaction_aware | independent | 12.421875 | 15.3125 | 13.4609375 | 1.5469698122081734 | completed |
| interaction_aware | standard_joint | 12.59375 | 15.3671875 | 13.4296875 | 1.5469698122081734 | completed |
| interaction_aware | interaction_aware | 12.4453125 | 15.28125 | 13.5078125 | 1.5469698122081734 | completed |
| interaction_aware | uniform3 | 12.3515625 | 15.1875 | 13.1953125 | 1.5469698122081734 | completed |

Lower perplexity is better. The matrix is post-hoc: it tests replayed
recombinations and does not prove causal mechanisms by itself.
