# ECA-bias diagnostic: A01

This controlled A01 run tested the ECA equation's explicit bias term while
keeping the v1.0.22-style candidate fixed: standard pooling and branch
Dropout(0.5), max-norm projection, no temporal-convolution BN, head ELU and
head Dropout(0.5), fusion enabled, and ECA enabled. It used seed 42,
deterministic CUDA, Adam at 0.001, batch size 64, 1000 epochs, and a fixed
230/58 split of the training Session.

The ECA 1D convolution used `bias=True`. The final internal validation
accuracy was **72.41%**; the highest internal validation accuracy was
**81.03% at epoch 298**. The target Session was not evaluated
(`test_evaluation=none`), and all losses and gradient norms were finite.

This did not improve the corresponding no-bias A01 diagnostic, so the ECA-bias
variant is not promoted to a nine-subject batch. The paper's ECA equation
contains a generic bias term, but the article does not provide source code or
enough implementation detail to establish whether the authors enabled the
convolution bias:
[AMEEGNet original article](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1540033/full).
