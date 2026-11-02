## Appendix B: Theoretical Analysis of Adaptive Entropy Gating (AEG)

In this section, we provide a formal theoretical justification for the Adaptive Entropy Gating (AEG) mechanism. We demonstrate how the dynamic weighting scheme achieves **information purification** by reducing the entropy of the motif distribution and enhancing the effective Signal-to-Noise Ratio (SNR) within a constrained representation bottleneck.

### B.1 Problem Formulation: Information Dilution and Bottleneck
Let a molecule be represented as a set of $K$ motifs $\mathcal{Z} = \{z_1, z_2, \dots, z_K\}$. Each motif is mapped to a $d$-dimensional embedding $\mathbf{h}_k \in \mathbb{R}^d$. In a standard graph pooling or attention mechanism, the global representation is often an aggregation: $\mathbf{H} = \sum_{k=1}^K \alpha_k \mathbf{h}_k$. 

**Constraint (Representation Bottleneck):** The embedding dimension $d$ is finite and fixed ($d \ll$ total structural complexity). When every motif is assigned equal importance ($\alpha_k = 1/K$), the task-relevant "signal" motifs $\mathcal{Z}_s$ are diluted by task-irrelevant "nuisance" motifs $\mathcal{Z}_N$, leading to semantic noise in the latent space.

### B.2 Entropy Reduction via Dynamic Gating
As implemented in Eq. (5), the AEG produces a normalized weighting distribution $\hat{\alpha}$ defined as a convex combination of the task-aware distribution $P$ and the uniform distribution $U$:
$$\hat{\alpha}_k = g \cdot P_k + (1-g) \cdot \frac{1}{K} \quad \text{where } g = 1 - E_{norm} \in [0, 1] \tag{B.1}$$
where $E_{norm}$ is the normalized Shannon entropy of the raw motif scores.

**Theorem 1 (Entropy Compression).** *The AEG mechanism guarantees that the entropy of the input distribution $H(\hat{\alpha})$ is strictly lower than that of the equally-weighted baseline $H(U)$, provided the gating factor $g > 0$ and $P \neq U$.*

*Proof.* In the discrete probability simplex $\Delta^K$, the uniform distribution $U = \{1/K, \dots, 1/K\}$ is the unique global maximum of the Shannon entropy function $H(\cdot)$, with $H(U) = \log K$. Since $P$ is derived from task-specific attention scores that identify key functional groups, it is inherently non-uniform ($H(P) < \log K$). 
By the definition of AEG, $\hat{\alpha}$ is a mixture that shifts the mass away from the maximum entropy point $U$ (uniform distribution) toward the task-informed distribution $P$ (from softmax). Due to the strict concavity of the entropy function, and the fact that $U$ is the unique maximizer:
$$H(\hat{\alpha}) = H(gP + (1-g)U) < H(U) = \log K$$
This entropy reduction signifies that AEG concentrates the model's "attention budget" on a sparse subset of motifs, effectively filtering out distributive noise. $\square$

### B.3 SNR Enhancement under Threshold $\tau$
We define the task-relevant signal $\mathcal{Z}_s = \{z_k \mid s_k > \tau\}$ and nuisance motifs $\mathcal{Z}_N = \{z_k \mid s_k \le \tau\}$, where $\tau$ is the learned significance threshold. The effective SNR of the aggregated representation is defined as the ratio of signal variance to noise variance.

**Lemma 1 (Variance Suppression).** *The AEG suppresses the contribution of nuisance motifs relative to the baseline.*

*Proof.* For any motif $z_k \in \mathcal{Z}_N$ whose task-relevance score $s_k$ is below the threshold $\tau$, the softmax-normalized probability $P_k$ is minimized. In the AEG distribution $\hat{\alpha}_k = g P_k + (1-g)/K$, the weight of such motifs satisfies $\hat{\alpha}_k < 1/K$ whenever $P_k < 1/K$. By decreasing the coefficients of motifs in $\mathcal{Z}_N$, AEG reduces the total variance contributed by background noise within the $d$-dimensional bottleneck, thereby increasing the effective signal purity for the LLM's reasoning tokens.  $\square$