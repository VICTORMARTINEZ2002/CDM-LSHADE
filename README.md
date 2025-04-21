# MSM-LSHADE

This repository contains the implementation of a new hybrid algorithm called **MSM-LSHADE** (**Multi-Shared Mining Success-History based Adaptive Differential Evolution with Linear Population Size Reduction**), which integrates **Data Mining** techniques and **Parallel Processing**.

MSM-LSHADE builds upon the **DM-LSHADE** algorithm [[1](#references)], a hybrid variant of **L-SHADE** [[2](#references)] with clustering-based data mining techniques. The DM-LSHADE introduced hybridization with **K-Means** [[3](#references)] and **X-Means** [[4](#references)] clustering algorithms, implemented using the [Pyclustering Library](https://github.com/annoviko/pyclustering).

In MSM-LSHADE, the hybrid algorithm is **parallelized using MPI (Message Passing Interface)**. The new algorithm is executed cooperatively by multiple LSHADE slave processes that **share mined patterns**, enabling dynamic population adaptation and knowledge exchange. These patterns are **mined and distributed by a master process**, which coordinates the interaction and pattern integration across slaves.

---

## References

[1] Santos, Raphael Gomes. *DM-LSHADE GitHub Repository*. ([GitHub](https://github.com/raphaelgoms/DM-L-SHADE))

[2] Tanabe, Ryoji and Alex S. Fukunaga. “Improving the search performance of SHADE using linear population size reduction.” *2014 IEEE Congress on Evolutionary Computation (CEC)* (2014): 1658–1665. ([PDF](https://ryojitanabe.github.io/pdf/tf-cec2014.pdf)) ([Code](https://ryojitanabe.github.io/code/LSHADE1.0.1_CEC2014.zip))

[3] R. Duda and P. Hart. *Pattern Classification and Scene Analysis*. John Wiley & Sons, 1973.

[4] Pelleg, D. and Moore, A. W. “X-means: Extending K-means with Efficient Estimation of the Number of Clusters.” In *Proceedings of the 17th International Conference on Machine Learning (ICML '00)*, 2000, pp. 727–734. ([PDF](https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf))
