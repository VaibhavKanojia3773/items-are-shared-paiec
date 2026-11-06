# Sparse hidden-like regime: K items per pair (eval pool K/2), 1 or 2 subjects per benchmark
B='"s_u":1.4,"s_t":0.5,"s_v":1.0,"s_w":2.2,"c":0.1,"n_cap":1.0,"s_f":0.7,"s_g":0.3,"a":0.4'
for K in 100 200; do for R in $K $((2*K)); do
  PAIEC_OUT=logbooks/runs python paiec/evaluate.py --subset data/proc/subset10k.parquet --tok both --template --formative $K --rows $R \
    --methods emp_mean,online0@g1,online0_f@g1,online0_f@g1c1,online_name_f@g1c1 --hp "{$B}" --tag SP_K${K}_R${R} 2>&1 | grep -A8 pooled | tail -n +2 | sed "s/^/[K=$K subj=$((R/K))] /"
done; done
