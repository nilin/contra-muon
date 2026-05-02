# Contra Muon

Contra Muon is a small modification of Muon: after forming Muon's Newton-Schulz
orthogonalized momentum update, subtract a fraction of the operator-normalized
momentum gradient:

```python
update = muon_update - contra_muon / 2 * operator_normalized_momentum_gradient
```

The subtraction is meant to temper the Muon direction by explicitly opposing the
raw gradient component in operator-norm units. In the initial NanoGPT Track 3
experiments, `contra_muon = 0.4`, so the subtracted component is `0.2` times the
operator-normalized momentum gradient.
