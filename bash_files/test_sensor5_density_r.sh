s=60



for r in {0.003,0.006,0.012,0.024,0.048,0.096,0.192,0.384,0.768}; do
    echo ${r}
    PARLAY_NUM_THREADS=${s} ./build/tests/density_query -r $r -i dataset/reallife/sensor5.dat -o out1 > results/densities/sensor5_density_$r.txt
done
