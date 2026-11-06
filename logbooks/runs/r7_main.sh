# Round 7 (codex review): one-response reveals, strong cheap baselines, v6/v7 text-free proxies, four regimes.
HP='{"a":0.4,"s_u":1.4,"s_t":0.5,"s_v":1.0,"s_w":2.2,"c":0.1,"n_cap":1.0,"s_f":0.7,"s_g":0.3}'
M=const,emp_mean,beta11,eb_bench,online0,online0@g1,online0_f@g1,online0_f,online0_f@g1c1,online_name_f@g1c1
run() { PAIEC_OUT=logbooks/runs python paiec/evaluate.py --subset data/proc/subset10k.parquet --tok both --template --single --no_provider --methods $M --hp "$HP" "$@"; }
run --tag R7_std > logbooks/runs/R7_std.log 2>&1 &
run --formative 100 --tag R7_high > logbooks/runs/R7_high.log 2>&1 &
run --formative 100 --rows 200 --tag R7_sp2 > logbooks/runs/R7_sp2.log 2>&1 &
run --formative 100 --rows 100 --tag R7_sp1 > logbooks/runs/R7_sp1.log 2>&1 &
wait
