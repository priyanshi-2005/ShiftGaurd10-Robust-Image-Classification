# ShiftGuard10: Robust Image Classification Under Distribution Shift

**WideResNet-28-10 trained from scratch with SGD, stochastic weight averaging, and test-time augmentation**

EE708: Fundamentals of Machine Learning · Indian Institute of Technology Kanpur  
Supervisor: **Prof. Rajesh K. Hegde**

**Aaditya Rathi** · **Anikeit Khanna** · **Priyanshi Agarwal** · **Shivesh Shukla** · **Jatin Kawatra**

---

## Problem statement

ShiftGuard10 is a 10-class image classification task. Each image is a 32×32 RGB picture, and the label is one of:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

The training set has **29,400** labeled images. The test set has **7,600** images. The score is **macro-averaged F1**: the unweighted mean of the per-class F1 scores. A model that is excellent on airplane and weak on truck is penalized as much as the reverse.

Two rules bound the solution. Training starts from random weights. No pretrained network and no data from outside this dataset may be used. The run has to be reproducible from the notebook alone, inside a Kaggle GPU session.

The practical question is:

**Can a model trained only on this set stay accurate when the test images do not look like the training images, and when some classes are almost absent?**

---

## Challenges

**The test distribution is shifted.** Train and test differ in brightness, contrast, and sharpness. The test images are brighter, higher in contrast, and slightly sharper on edges. A network that memorizes the training appearance misses classes once those surface statistics change.

**The classes are severely imbalanced.** Four classes have 5,000 images each. Truck has 100. That is a **50×** gap. The full counts are:

| Class | Training images |
| --- | ---: |
| airplane | 5,000 |
| automobile | 5,000 |
| bird | 5,000 |
| cat | 5,000 |
| dog | 4,000 |
| deer | 4,000 |
| frog | 500 |
| horse | 500 |
| ship | 300 |
| truck | 100 |

Macro F1 weights every class equally, so the 100 truck images matter as much as the 5,000 airplane images.

**Pixels alone do not separate the classes.** PCA and t-SNE on raw pixels show no clean class clusters. The decision boundary has to come from a deep representation, built at the native 32×32 resolution. There is no room to resize into a large ImageNet backbone, and those backbones are disallowed anyway.

**The learning rate and the batch size have to move together.** SGD with learning rate 0.1 is the right setting at batch size 512. The same learning rate at batch size 128 is about four times too large, and training diverges. This is the linear scaling rule: learning rate divided by batch size stays constant.

**Time is limited.** The notebook runs on one Kaggle T4 with a session cap of about 9 hours. The submitted run is a single seed (`42`), 300 epochs, about **5 hours 38 minutes**.

---

## How we solved it

The pipeline is five pieces that only work together. Each one answers one of the problems above.

**1. WideResNet-28-10, trained from scratch.** Depth 28, widening factor 10, about 36.5 million parameters. Channel widths go 16 → 160 → 320 → 640. Dropout is 0.3. Weights use Kaiming He initialization, which keeps the first epochs from stalling in a network this deep. The model reads 32×32 images directly. Images are normalized with CIFAR-10 statistics: mean `(0.4914, 0.4822, 0.4465)`, standard deviation `(0.2470, 0.2435, 0.2616)`.

**2. Two imbalance corrections at once.** A square-root-inverse `WeightedRandomSampler` makes rare classes show up in batches without letting the 100 truck images dominate every step. `BalancedSoftmax` then shifts the logits by `log(class frequency)` before cross-entropy, so the decision boundaries match the true frequencies. Label smoothing is 0.1. At inference the loss is unused, so the correction does not have to be undone by hand.

**3. Augmentation that imitates the shift.** Training uses a random crop with padding 4, a horizontal flip, CIFAR-10 AutoAugment, and a 16×16 Cutout patch. On half of the batches, MixUp or CutMix is chosen with equal probability (α = 1.0). Cutout and CutMix force the model to use more than one local region. Color and crop jitter make the training batches less tied to one brightness and contrast.

**4. SGD, a cosine schedule, then stochastic weight averaging.** Optimizer: SGD, momentum 0.9, Nesterov, weight decay `5e-4`, batch size 512, learning rate 0.1, mixed precision, gradient clip 5.0. The learning rate warms up for 5 epochs, then cosine-decays until epoch 240. From epoch 241 to 300, SWA averages the weights at a constant learning rate of 0.005. Batch-norm statistics are re-estimated on the training set after averaging. SWA lifted validation macro F1 from **0.8618** at the best single checkpoint (epoch 225) to **0.8667**.

**5. Thirty-view test-time augmentation.** Each test image is scored once clean and 29 more times under random crop, horizontal flip, mild color jitter, and a small rotation (±10°). The 30 softmax outputs are averaged. Prediction variance falls sharply by 30 views and barely moves after that, so 30 is the operating point that fits the time budget.

Validation is a stratified 5% holdout (1,470 images) from the same seed. The public leaderboard score uses the 30-view predictions on all 7,600 test images.

A second or third seed, with logits averaged across seeds, is wired in the notebook (`seeds`) and was left as a single seed because of the session limit. The report estimates that a multi-seed ensemble would move the leaderboard F1 above 0.95.

---

## Results

| | |
| --- | --- |
| Validation macro F1 (SWA, seed 42) | 0.8667 |
| Public leaderboard macro F1 | 0.930892 |
| Runtime | 5h 38m 17s on one T4 |
| Epochs | 300 |
| TTA views | 30 |

Validation macro F1 is lower than the leaderboard score because the holdout is small for the tail classes (5 truck images, 15 ship images) and because TTA is applied on the test set. Well-represented classes are near ceiling on the holdout: automobile F1 is 1.00 and airplane is 0.99. Truck validation F1 is 0.28, which is what five positive examples can support. The leaderboard gap above the validation number is the evidence that TTA is closing the shift.

Per-class validation precision, recall, and F1 are in `ShiftGuard10_Report.pdf` and on the results slide of `ShiftGuard10_Presentation.pdf`.

---

## Files in this repository

```text
.
├── README.md
├── ShiftGuard10_Report.pdf
├── ShiftGuard10_Presentation.pdf
├── code/
│   └── shiftguard10.ipynb
└── dataset/
    ├── classes.txt
    ├── train_labels.csv
    ├── sample_submission.csv
    ├── train_images/          # 29,400 PNG files
    └── test_images/           # 7,600 PNG files
```

**`ShiftGuard10_Report.pdf`**  
Three-page project report. It states the problem, the dataset findings (imbalance, pixel statistics, train–test shift, and the failure of linear features), the WideResNet pipeline, the training configuration, per-class validation metrics, and the public leaderboard score.

**`ShiftGuard10_Presentation.pdf`**  
Ten-slide talk used for the course presentation: problem, exploratory analysis, the five-part pipeline, the learning-rate schedule, augmentation, imbalance handling, the logged training results, and the takeaways.

**`code/shiftguard10.ipynb`**  
The Kaggle notebook that produced the submission. It contains the dataset class, Cutout, BalancedSoftmax, the square-root-inverse sampler, WideResNet-28-10, MixUp and CutMix, the warmup–cosine–SWA loop, 30-view TTA, and the code that writes `submission.csv`. It was executed on Kaggle, so `DATA_DIR` points at

`/kaggle/input/competitions/shift-guard-10-robust-image-classification-challenge`

To run against this repository, point `DATA_DIR` at the local `dataset/` folder. `train_labels.csv` and `sample_submission.csv` use the same column names the notebook expects (`id`, `label`). Image files are named `{id}.png` with a zero-padded 6-digit id, which is what `ShiftGuardDataset` loads.

**`dataset/classes.txt`**  
The ten class names, one per line, in the index order used by the model.

**`dataset/train_labels.csv`**  
29,400 rows. Columns: `id`, `label`.

**`dataset/sample_submission.csv`**  
7,600 rows. Columns: `id`, `label`. Every label in the file is the placeholder `airplane`. The notebook reads it for the test ids and overwrites the labels with its predictions.

**`dataset/train_images/`**  
29,400 PNG images, 32×32 RGB, ids matching `train_labels.csv`.

**`dataset/test_images/`**  
7,600 PNG images, 32×32 RGB, ids matching `sample_submission.csv`. These images have no public labels.
