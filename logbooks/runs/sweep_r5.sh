# Round 5 local sweep: nested name effect, 2-config ensemble, robustness to pool size (online0_f = final minus redundant propagation)
V5='"a":0.4,"s_u":1.4,"s_t":0.5,"s_v":1.0,"s_w":2.2,"c":0.1,"n_cap":1.0,"s_f":0.7,"s_g":0.3'
HP2='{"a":0.6,"s_v":0.7,"s_w":1.6,"c":0.2,"s_f":0.5,"s_g":0.0}'
run() { PAIEC_OUT=logbooks/runs python paiec/evaluate.py --subset data/proc/subset10k.parquet --tok both --template "$@" 2>&1 | grep -A30 pooled | tail -n +2 | sed "s/^/[$TAG] /"; }
for REG in "--formative 100" ""; do
  TAG="R5name$( [ -n "$REG" ] && echo F || echo S)"
  for sn in 0.3 0.6 1.0; do TAG="${TAG%_*}_sn$sn" run $REG --methods online0_f@g1c1,online0_f+n@g1c1 --hp "{$V5,\"s_n\":$sn}" --tag "${TAG%_*}_sn$sn"; done
  TAG="R5ens$( [ -n "$REG" ] && echo F || echo S)" run $REG --methods online0_f+e@g1c1 --hp "{$V5}" --hp2 "$HP2" --tag "R5ens$( [ -n "$REG" ] && echo F || echo S)"
done
for K in 50 200; do TAG="R5K$K" run --formative $K --methods online0@g1,online0_f@g1,online0_f@g1c1,online0_f@g1c2 --hp "{$V5}" --tag R5K$K; done
