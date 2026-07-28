# Core philosophy: identity, relation, and verified meaning

## 1. The central thesis

The method is built on one principle:

> Preserve the part of a local representation that expresses identity, write information only into relational degrees of freedom, and use the preserved identity as a witness when deciding which recovered relations are trustworthy.

This is not an analogy added after the algorithm. It determines the carrier, the embedding law, the decoder, and the application metric.

## 2. Identity–relation decomposition

For every 8×8 DCT block, six selected coefficients form the strict-upper vector

\[
n=(t_{12},t_{13},t_{14},t_{23},t_{24},t_{34})^{\mathsf T}\in\mathbb R^6,
\]

while four disjoint coefficients define the diagonal

\[
D=\operatorname{diag}(\lambda_1,\lambda_2,\lambda_3,\lambda_4).
\]

Together they define a virtual upper-triangular Schur form

\[
T=D+N.
\]

The diagonal contains the eigenvalues of \(T\). It therefore represents the local spectral identity. The strict-upper part \(N\) represents directed couplings among those spectral components: the relational freedom.

The algorithm never changes \(D\). It modifies only \(N\). Consequently,

\[
\operatorname{spec}(T^\star)=\operatorname{spec}(T),\qquad
\operatorname{tr}(T^\star)=\operatorname{tr}(T),\qquad
\det(T^\star)=\det(T).
\]

## 3. Minimum-change inscription

Let \(H\in\mathbb R^{3\times 6}\) have orthonormal rows and let \(t\in\mathbb R^3\) be the nearest QIM parity target. The embedding is

\[
n^\star=n+H^{\mathsf T}(t-Hn).
\]

This is the unique minimum-Euclidean update satisfying \(Hn^\star=t\). Because the vector consists of strict-upper entries, it is also the minimum-Frobenius update of the virtual Schur matrix.

The payload-wise coset selector compares two physically equivalent parity labels and chooses the lower-energy one:

\[
f_j^\star=\arg\min_{f\in\{0,1\}}
\sum_{r,k}\bigl[Q_{b_j\oplus f}(c_{r,k,j})-c_{r,k,j}\bigr]^2.
\]

This reduces distortion without decreasing the parity spacing.

## 4. Identity as a decoding witness

The same diagonal-derived scale that embedding leaves unchanged is stored as a compact key reference. At extraction, global gain is separated from local damage:

\[
r_i=\log\frac{s_i'}{s_i},\qquad
\widetilde r_i=r_i-\operatorname{median}_j r_j.
\]

The local trust weight is

\[
\omega_i=\omega_{\min}+(1-\omega_{\min})
\exp\!\left[-\frac12\left(\frac{\widetilde r_i}{\sigma_w}\right)^2\right].
\]

A global brightness or contrast change shifts most blocks together and is removed by the median. Local occlusion, erasure, or corruption produces an outlying witness drift and is downweighted. Thus the protected identity is not passive; it later judges the reliability of the writable relations.

## 5. Meaning rather than resemblance

A binary-logo NC score remains useful for research diagnosis, but it is not the final acceptance rule. Real payloads are accepted only when they pass:

\[
\text{convolutional decoding}
\land \text{CRC validation}
\land \text{digital-signature verification}.
\]

The application therefore asks whether an exact signed statement survived, not merely whether a recovered pattern resembles the original watermark.

## 6. Research contribution

The core contribution is the closed loop:

1. **Preserve identity** by leaving Schur diagonal coordinates unchanged.
2. **Write relations** through an orthogonal minimum-Frobenius parity projection.
3. **Use identity as witness** to gate damaged relational evidence.
4. **Verify meaning** through channel coding, CRC, and signatures.

This connects linear-algebraic invariance, robust communication, and media provenance in one method rather than presenting them as unrelated additions.
