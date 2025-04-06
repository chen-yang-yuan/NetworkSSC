# NetworkSSC

## NetworkSSC: Network-Guided Sparse Subspace Clustering on Single-Cell Data

#### Chenyang Yuan, Shunzhou Jiang, Songyun Li, Jicong Fan* and Tianwei Yu*

With the rapid development of single-cell RNA sequencing (scRNA-seq) technology, researchers can now investigate gene expression at the individual cell level. Identifying cell types via unsupervised clustering is a fundamental challenge in analyzing single-cell data. However, due to the high dimensionality of expression profiles, traditional clustering methods often fail to produce satisfactory results. To address this problem, we developed NetworkSSC, a network-guided sparse subspace clustering (SSC) approach. NetworkSSC operates on the same assumption as SSC that cells of the same type have gene expressions lying within the same subspace. Additionally, it integrates a regularization term incorporating the gene network’s Laplacian matrix, which captures functional associations between genes. Comparative analysis on nine scRNA-seq datasets shows that NetworkSSC outperforms traditional SSC and other unsupervised methods in most cases.

![NetworkSSC workflow](docs/workflow.png)<br>