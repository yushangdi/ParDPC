s=60


for r in {0.3,0.6,1.2,2.4,4.8}; do
    echo ${r}
    PARLAY_NUM_THREADS=${s} ./build/tests/density_query -r $r -i dataset/reallife/HT.pbbs -o out1 > results/densities/HT_density_$r.txt
done

