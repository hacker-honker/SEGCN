# SEGCN: Structural Enhancement Graph Clustering Network
We propose a Structural Enhancement Graph Clustering Network(SEGCN) to learn the graph topology information hidden in the dependencies of nodes attribute and the global structure information.

We have added comments in the code, the specific details can correspond to the explanation in the paper.

# Environment
+ Python[3.6.12]
+ Pytorch[1.7.1]
+ GPU (GeForce RTX 2080 Ti) & (NVIDIA GeForce RTX 3090) & (Quadro RTX 8000)

# Hyperparameters
+ The learning rates of USPS, HHAR, and ACM datasets are set to 0.001, and the learning rates of CiteSeer dataset are set to 0.0001. lambda1 and lambda2 are set to {10, 100} for USPS, {1, 0.1} for HHAR, {0.1, 0.01} for ACM and CiteSeer datasets.

# To run code
+ Step 1: set the hyperparameters for the specific dataset;
+ Step 2: python SEGCN.py --name [data_name]
* For examle, if you would like to run SEGCN on the usps dataset, you need to
* first set {10, 100} for USPS;
* then run the command "python SEGCN.py --name usps"
