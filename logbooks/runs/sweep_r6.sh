# Round 6: name-prior strength (a) and name-pooled subject offset, both regimes, tokens on, coverage b=1
B='"s_u":1.4,"s_t":0.5,"s_v":1.0,"s_w":2.2,"c":0.1,"n_cap":1.0,"s_f":0.7,"s_g":0.3'
for R in "--formative 100" ""; do T=$([ -n "$R" ] && echo F || echo S)
  for a in 0.2 0.6 0.8 1.0; do
    PAIEC_OUT=logbooks/runs python paiec/evaluate.py --subset data/proc/subset10k.parquet --tok both --template $R --methods online_name_f@g1c1 --hp "{$B,\"a\":$a}" --tag R6a${a}$T 2>&1 | grep -A2 pooled | tail -1 | sed "s/^/[$T a=$a] /"
  done
done
