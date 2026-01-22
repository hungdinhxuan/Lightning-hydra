
## Distil XLSR-ConformerTCM v2 (exp)

```bash
./scripts/benchmark.sh -g 3 -c distil/jit_inference -b data/cnsl_benchmark -m /nvme1/hungdx/pretrained_unifed_kd/X_5_Conf-TCM_pruned05_c_m_jit.pt -r logs/results/cnsl_benchmark -n "X_5_Conf-TCM_pruned05_c_m_jit" -l false -t 32000
```