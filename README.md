# gini_coefficient
Calculates all-versus-all similarities of np.ndarrray (n x d) of embeddings. Calculation of Gini coefficients (output: n x 1 vector) from n x d array of L2-normalized embeddings for assessment of similarity across all n-elements. 

*NOTE:* code assumes all embeddings are **L2-NORMALIZED**.

<P align="center">
<IMG SRC="assets/example.png" CLASS="center" ALT="illustration of Gini coefficient calculation for all-vs-all similarity scores for embeddings">
</P>
<P>

### Example

```python
giniCoefficient(embeddings: np.ndarray, normalize: bool=True, memory_efficient: bool=True)
```

**Input:**

**embeddings** : `np.ndarray` of n x d dimensions *NOTE:* code assumes all embeddings are **L2-NORMALIZED**

Enabling the `normalize` feature results in `MinMax` scaling of outputs such that [0,1] for all Gini values.

Enabling the `memory_efficient` feature results in memory-efficient, but potentially less time-efficient,
calculation of the Gini coefficient for each element in the input array.

**Output:**

**array** : `np.ndarray` of n x 1 dimensions of the Gini coefficients of the input array

##
This is a `numpy` implementation of: ["Gini Coefficient as a Unified Metric for Evaluating Many-versus-Many Similarity in Vector Spaces"](https://arxiv.org/abs/2411.07983). If using more than 1,000 or 10,000 embeddings/vectors, it is recommended that this code should be implemented with [CuPy](https://cupy.dev/) or [PyTorch](https://pytorch.org/) for acceleration on GPUs. The similarity calculation is the slow step and this is calculation is greatly accelerated on GPU infrastructure (<1 sec for 1e10 elements on GPU vs. hours on CPU).

Dependencies include `numpy` and `scikit-learn`

Code uses the bootstrap method for calculating a Gini coefficient, as described in:
1) Dixon, P. M.; Weiner, J.; Mitchell-Olds, T.; Wooley, R. "Bootstrapping the Gini Coefficient of Inequality." *Ecology* **1987**, *68*, 1548-1551.
2) Dixon, P. M.; Weiner, J.; Mitchell-Olds, T.; Wooley, R. "Errratum to Bootstrapping the Gini Coefficient of Inequality." *Ecology* **1988**, *69*, 1307.
