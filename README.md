# Contra Muon

Contra Muon is a small modification of Muon: after forming Muon's Newton-Schulz
orthogonalized momentum update, subtract a fraction of the operator-normalized
momentum gradient:

```python
update = muon_update - contra_muon / 2 * operator_normalized_momentum_gradient
```

In the initial NanoGPT Track 3
experiments, `contra_muon = 0.4`, so the subtracted component is `0.2` times the
operator-normalized momentum gradient.

## Reasoning

```
Let sum_i s_i u_i v_i' be the SVD of the gradient (estimated by the momentum vector). suppose the update is sum_i a_i m_i where m_i=u_i v_i' are the singular modes. then c_i m_i contributes ~ a_i s_i to the loss delta. So in (momentum) SGD, larger singular directions contribute quadratically more to the loss delta. But even in Muon, larger singular directions constribute more, but only linearly. Contra-Muon with coefficient 1 makes the large singular directions contribute the same amount to the loss delta, to first order. This is because a_i=1-r_i/2 where r_i=1 is the i'th singular value relative to the largest. So the contribution is proportional to f(r_i) = r_i*(1-r_i/2) = r_i - r_i^2/2 which has f'(1)=1, i.e. it is approximately constant near the top singular value where r_i~1.
```
