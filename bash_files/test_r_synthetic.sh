s=60
R=10000000
d=100
#for r in {200,300,450,675,1012.5,1518.75,}; do
for r in {2278.125,3417.1875,5125.78125}; do
    echo ${r}
    PARLAY_NUM_THREADS=${s} ./build/tests/density_query -r $r -i dataset/simden/${R}.txt -o out1 > results/densities/simden_density_${r}.txt
    PARLAY_NUM_THREADS=${s} ./build/tests/density_query -r $r -i dataset/varden/${R}.txt -o out1 > results/densities/varden_density_${r}.txt
    PARLAY_NUM_THREADS=${s} ./build/tests/density_query -r $r -i dataset/uniform/${R}.txt -o out1 > results/densities/uniform_density_${r}.txt
done

