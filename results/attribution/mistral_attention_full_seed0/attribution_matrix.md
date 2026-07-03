# Joint Compression Attribution Matrix

| Depth source | Quant source | WikiText2 PPL | C4 PPL | FineWeb-Edu PPL | Compression | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| independent | independent | 13.171875 | 16.046875 | 14.21875 | 1.5469698122081734 | completed |
| independent | standard_joint | 13.328125 | 16.296875 | 14.3828125 | 1.5469698122081734 | completed |
| independent | interaction_aware | 14.953125 | 18.21875 | 15.984375 | 1.5469698122081734 | completed |
| independent | uniform3 | 13.1953125 | 16.296875 | 14.2421875 | 1.5469698122081734 | completed |
| standard_joint | independent | 13.3828125 | 16.65625 | 14.75 | 1.5469698122081734 | completed |
| standard_joint | standard_joint | 13.25 | 16.65625 | 14.8359375 | 1.5469698122081734 | completed |
| standard_joint | interaction_aware | 15.578125 | 20.328125 | 17.28125 | 1.5469698122081734 | completed |
| standard_joint | uniform3 | 13.3515625 | 16.84375 | 14.8125 | 1.5469698122081734 | completed |
| interaction_aware | independent | 12.375 | 15.015625 | 13.1484375 | 1.5469698122081734 | completed |
| interaction_aware | standard_joint | 12.8671875 | 15.609375 | 13.3828125 | 1.5469698122081734 | completed |
| interaction_aware | interaction_aware | 12.640625 | 15.4609375 | 13.6171875 | 1.5469698122081734 | completed |
| interaction_aware | uniform3 | 12.4921875 | 15.3671875 | 13.3046875 | 1.5469698122081734 | completed |

Lower perplexity is better. The matrix is post-hoc: it tests replayed
recombinations and does not prove causal mechanisms by itself.
