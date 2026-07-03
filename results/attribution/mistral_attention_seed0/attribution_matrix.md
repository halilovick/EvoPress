# Joint Compression Attribution Matrix

| Depth source | Quant source | WikiText2 PPL | C4 PPL | FineWeb-Edu PPL | Compression | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| independent | attention_independent | 13.171875 | 16.046875 | 14.21875 | 1.5469698122081734 | completed |
| independent | attention_joint | 13.328125 | 16.296875 | 14.3828125 | 1.5469698122081734 | completed |
| independent | uniform3 | 13.1953125 | 16.296875 | 14.2421875 | 1.5469698122081734 | completed |
| attention_joint | attention_independent | 13.3828125 | 16.6875 | 14.6953125 | 1.5469698122081734 | completed |
| attention_joint | attention_joint | 13.25 | 16.65625 | 14.8359375 | 1.5469698122081734 | completed |
| attention_joint | uniform3 | 13.3515625 | 16.84375 | 14.8125 | 1.5469698122081734 | completed |

Lower perplexity is better. The matrix is post-hoc: it tests replayed
recombinations and does not prove causal mechanisms by itself.
