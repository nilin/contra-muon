# Contra-Muon and Power-Muon

Nilin


Contra-Muon and Power-Muon (p<0) are exaggerations of Muon which further boost small singular values or apply damping to large singular values of the gradient. The goal is to compensate for the smaller leverage of small singular directions to boost diversity in training. 


## Background

[Muon](https://kellerjordan.github.io/posts/muon/) modifies the momentum gradient of matrix-shaped weights by making all singular values close to 1, thereby boosting the small singular modes. Muon is not only a definition in terms of linear algebra, but also an algorithm which calculcates the update efficiently using Newton-Schultz iteration.


## Contra-Muon

Contra Muon is a small modification of Muon: after forming Muon's Newton-Schulz orthogonalized momentum update, subtract a fraction of the operator-normalized momentum gradient:

```python
update =  (1 + contra_muon_coeff) * muon_update - contra_muon_coeff * operator_normalized_momentum_gradient
```

where `0 < contra_muon_coeff <= 1`.


## Power-Muon

The Newton-Schultz iterates in Muon produce approximations to `f(g)` where `g` is the matrix-shaped gradient, `f(g)` is shorthand for `Uf(D)V` where `UDV` is the SVD of `g`. Here `f` is a function `f(0)=0`, `f((eps,1])=1` where `eps` gets smaller with each iteration. While Muon normally uses the last iterate as an approximation to UV, we can also take linear combinations of the previous iterates to compute other functions of `g`. Contra-Muon can be considerd a special case where we use the 0'th and last iterate. In particular we are interested in power functions `x^p` where `-1<=p<1`. Standard Muon corresponds to `p=0`.



## Reasoning for boosting small singular values beyond Muon

Let

```text
G = sum_i s_i u_i v_i^T
```

be the SVD of the gradient, estimated in practice by the momentum buffer. Suppose the update is

```text
U = sum_i a_i m_i,    where m_i = u_i v_i^T.
```

Then the `a_i m_i` component contributes approximately `a_i s_i` to the first-order loss change. In momentum SGD, larger singular directions therefore contribute quadratically more to the loss change. In Muon, larger singular directions still contribute more, but only linearly.

Contra Muon with coefficient `1` makes the largest singular directions contribute approximately the same amount to the loss change, to first order. If

```text
r_i = s_i / s_1,
```

then Contra Muon uses

```text
a_i = 2 - r_i.
```

The contribution is therefore proportional to

```text
f(r_i) = r_i * (2 - r_i) = 2r_i - r_i^2.
```

Since `f'(1) = 0`, this contribution is approximately flat near the top singular
value, where `r_i ~= 1`.

## Results
As a proof of concept I used Contra-Muon in modded-nanogpt track 3: https://github.com/KellerJordan/modded-nanogpt/pull/275, producing a record run.
