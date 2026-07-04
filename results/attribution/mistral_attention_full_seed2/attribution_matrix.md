# Joint Compression Attribution Matrix

| Depth source | Quant source | WikiText2 PPL | C4 PPL | FineWeb-Edu PPL | Compression | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| independent | independent | 12.3515625 | 15.1015625 | 13.6171875 | 1.5469698122081734 | completed |
| independent | standard_joint | 12.3984375 | 15.015625 | 13.6171875 | 1.5469698122081734 | completed |
| independent | interaction_aware | 13.1953125 | 15.828125 | 14.3828125 | 1.5469698122081734 | completed |
| independent | uniform3 | 12.1796875 | 15.015625 | 13.4296875 | 1.5469698122081734 | completed |
| standard_joint | independent | 12.9140625 | 15.3125 | 13.5078125 | 1.5469698122081734 | completed |
| standard_joint | standard_joint | 13.015625 | 15.3125 | 13.5390625 | 1.5469698122081734 | completed |
| standard_joint | interaction_aware | 13.671875 | 15.828125 | 14.3828125 | 1.5469698122081734 | completed |
| standard_joint | uniform3 | 12.7890625 | 15.1640625 | 13.4609375 | 1.5469698122081734 | completed |
| interaction_aware | independent | 11.4453125 | 14.3828125 | 12.4453125 | 1.5469698122081734 | completed |
| interaction_aware | standard_joint | 11.4453125 | 14.4375 | 12.546875 | 1.5469698122081734 | completed |
| interaction_aware | interaction_aware | 12.1796875 | 15.015625 | 13.2265625 | 1.5469698122081734 | completed |
| interaction_aware | uniform3 | 11.375 | 14.4140625 | 12.3984375 | 1.5469698122081734 | completed |

Lower perplexity is better. The matrix is post-hoc: it tests replayed
recombinations and does not prove causal mechanisms by itself.
