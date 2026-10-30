# IITK Round One

Submissions for the two problem statements of round one. Each lives in its own
directory with its own README, code and deliverables.

| | problem | status |
| --- | --- | --- |
| [`PS-1/`](PS-1/) | **Backdoor detection from hardware performance counters** | complete |
| [`PS-2/`](PS-2/) | **Black-box adversarial attack on a CNN** | complete |

## PS-1 — Backdoor detection from hardware performance counters

A one-class detector for backdoor-triggered inferences. The training data is
800 HPC traces (cache-references, cycles, LLC-loads) from clean inferences
only, so the detector learns what clean traces look like and flags the ones
that don't fit. It is a three-component Gaussian mixture on the standardised
counters, scored by negative log-density, with the threshold set on held-out
clean traces.

| | |
| --- | --- |
| **Model** | Gaussian mixture, 3 components, full covariance |
| **Threshold** | 4.664, calibrated to a **10 %** clean false-positive rate |
| Clean FPR on held-out traces | 0.049 at a 0.05 target (20 random splits) |
| **AUROC, real backdoor traces** | **0.865** (z-score, Mahalanobis, PCA, one-class SVM: 0.46 to 0.52) |
| AUROC, synthetic counter shifts | 0.858 mean, 0.767 for the hardest direction |
| Training traces removed | none |

The training set has no backdoor traces, so the hidden test score can't be
measured locally. The real-backdoor AUROC comes from the public traces of
Alam et al. (DATE 2024), which use different counters and hardware, and is
there to compare methods. See [`PS-1/README.md`](PS-1/README.md) for usage and
[`PS-1/reports/report.pdf`](PS-1/reports/report.pdf) for the written report.

![PS-1 clean traces and decision boundary](PS-1/reports/figures/fig_data.png)

## PS-2 — Black-box adversarial attack on a CNN

A decision-based (**hard-label**) attack on the provided CIFAR-10 CNN: the
model may be queried at arbitrary inputs and only its predicted label read —
no architecture, parameters, logits or intermediate outputs. One adversarial
`.png` is produced for each of the 1000 clean images.

| | |
| --- | --- |
| **Attack success rate** | **100.00 %** (1000/1000) |
| **PSNR** | mean **49.00 dB** |
| **SSIM** | mean **0.99762** |
| L2 per image (0-255 scale) | mean 59.0, median 58.3 |
| L-inf per image (0-255 scale) | mean 6.9, median 7 |
| Model inferences | **16,900 per image** |

Measured by reading the submitted `.png` files back from disk and querying the
provided model. See [`PS-2/README.md`](PS-2/README.md) for the method and
[`PS-2/report/report.pdf`](PS-2/report/report.pdf) for the written report.

![PS-2 examples](PS-2/report/examples.png)
