# Contra-Muon and Power-Muon

Nilin


Contra-Muon and Power-Muon are exaggerations of Muon: 


## Background

[Muon](https://kellerjordan.github.io/posts/muon/) modifies the momentum gradient by making all singular values close to 1, thereby boosting the small singular modes.


## Contra-Muon

Contra Muon is a small modification of Muon: after forming Muon's Newton-Schulz orthogonalized momentum update, subtract a fraction of the operator-normalized momentum gradient:

```python
update =  (1 + contra_muon_coeff) * muon_update - contra_muon_coeff * operator_normalized_momentum_gradient
```

where `0 < contra_muon_coeff <= 1`.

## Reasoning

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
a_i = 1 - r_i / 2.
```

The contribution is therefore proportional to

```text
f(r_i) = r_i * (1 - r_i / 2) = r_i - r_i^2 / 2.
```

Since `f'(1) = 0`, this contribution is approximately flat near the top singular
value, where `r_i ~= 1`.

## Results
As a proof of concept I used Contra-Muon in modded-nanogpt track 3: https://github.com/KellerJordan/modded-nanogpt/pull/275, producing a record run.
