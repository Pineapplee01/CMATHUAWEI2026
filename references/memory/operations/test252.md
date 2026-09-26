# Aligned Test 252 Protocol

The six baseline methods use the same frozen aligned test protocol as
problem2_retrain_v2.

- Seven modality combinations: T, A, V, TA, TV, AV and TAV.
- Six span rates: 10%, 20%, 30%, 40%, 50% and 70%.
- Start, middle and end each use mask seed 2026.
- Random position uses mask seeds 2026, 2027 and 2028.

The protocol has 252 scenarios per model seed. Every scenario contains all 727
aligned test samples. Scenario results are stored inside the verified run under
test252. Execution remains paused until baseline training is resumed.
