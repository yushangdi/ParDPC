# ParCluster

Repository for parallel clustering through Density-Peak Clustering

Need to change the dimension in `parameter.h` before compilation.

./tests/dpc_priority -r 6 -i ../../data/gaussian_4_10000_128.data 

./tests/dpc_alternate -k 6 -d 251 -i ../../data/gaussian_4_10000_128.data -o /home/ubuntu/DPC-ANN/results/gaussian_4_10000_128_priority.cluster -decision  /home/ubuntu/DPC-ANN/results/gaussian_4_10000_128_priority.dg


./src/dpc_sddp -i ../data/gaussian_example/gaussian_4_1000.data

./src/dpc_sddp -i ../data/mnist/mnist.txt


TODO:
- make pskdtree split at median
- add option to tree: use spatial or object median split
- change read density back to compute density