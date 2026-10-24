# MNIST Digit Classification — Project Report

A PyTorch Lightning pipeline for the Kaggle "Digit Recognizer" (MNIST) dataset. The
required deliverable is a multi-layer perceptron (MLP): clean and split the data
properly, build a pipeline where the architecture and hyperparameters are easy to
change, systematically compare across optimizers, batch sizes, activation functions,
learning rates, dropout ratios, hidden-layer architectures, and loss functions,
evaluate the best configuration once on held-out test data, and propose further
improvements. A convolutional neural network (CNN) was added afterward as a
self-directed stretch goal, built and searched with the same rigor as the MLP.

**Results at a glance:**

| | Validation accuracy (balanced) | Test accuracy (balanced) | Test precision | Test recall |
|---|---|---|---|---|
| MLP baseline | 97.19% | — | — | — |
| **MLP, tuned** | 98.12% | **98.30%** | 98.29% | 98.30% |
| CNN baseline (untuned) | 98.71% | — | — | — |
| **CNN, tuned** | 99.22% | **99.21%** | 99.22% | 99.21% |

All accuracies in this report that come from the training pipeline are **balanced
accuracy**: torchmetrics' `MulticlassAccuracy` with macro averaging (its default,
now set explicitly in `src/model.py`) averages the per-class accuracies, which is
mathematically identical to macro recall, hence the matching columns. Plain accuracy
(correct ÷ total) on the test set is 98.31% (MLP) and 99.23% (CNN); on these
near-balanced classes the two differ by at most 0.02 points. The distribution-shift
evaluation (section 6.2) reports plain accuracy.

The CNN's *untuned* baseline (98.71%) already beats the MLP's *fully tuned* best
(98.12%) — a direct illustration of architecture mattering more than hyperparameter
tuning for this problem. The tuned CNN roughly halves the MLP's error rate (1.70% →
0.79% of test digits misclassified).

Validation accuracy is measured on data that was never trained on, but which *was*
used to choose the best hyperparameters — so it runs slightly optimistic. Test
accuracy comes from data touched exactly once, after every decision was already
locked in, which is why it's the number reported as the final result.

---

## 1. Data

### 1.1 Source

Kaggle's [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)
competition. It ships `train.csv` (42,000 labeled rows) and `test.csv` (28,000 rows).
`test.csv` has **no label column** — confirmed by inspection before writing any split
logic — since it exists purely for Kaggle's own leaderboard scoring, not for local
evaluation. The entire 60/20/20 split used in this project is therefore built from
`train.csv` alone.

### 1.2 Cleaning

Five checks run automatically every time the data loads — a permanent part of the
pipeline, not a one-off notebook pass:

| Check | Result |
|---|---|
| Shape | 42,000 rows × 785 columns (784 pixels + label) |
| Missing values | 0 |
| Exact duplicate rows | 0 |
| Pixel values outside [0, 255] | 0 |
| Label values outside [0, 9] | 0 |
| Blank (all-zero) images | 0 |

Every check came back clean. That's not "nothing to report" — it's five specific,
real failure modes checked for explicitly instead of trusting a well-known dataset
blindly. If this pipeline were ever pointed at messier data, these same checks would
engage for real.

### 1.3 Class balance

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Count | 4132 | 4684 | 4177 | 4351 | 4072 | 3795 | 4137 | 4401 | 4063 | 4188 |
| % of dataset | 9.84 | 11.15 | 9.95 | 10.36 | 9.70 | 9.04 | 9.85 | 10.48 | 9.67 | 9.97 |

![Class balance](results/eda_class_balance.png)

Range 9.04%–11.15% — close enough to uniform that class-imbalance handling (weighted
sampling, class-weighted loss, focal loss) was correctly judged unnecessary. This was
a measured conclusion, verified from the data, not a default assumption.

**Why this matters:** a model trained on lopsided data can score high accuracy just by
leaning on the majority class. Confirming balance up front is what justifies using
accuracy as a trustworthy headline metric here. The accuracy the pipeline reports is
in fact macro-averaged (balanced) accuracy, as are precision and recall, since that's
the safer default regardless of what the balance check finds; with balanced classes it
agrees with plain accuracy to within 0.02 points.

A visual sanity check beyond the counts: one real sample image per digit class,
straight from the raw pixel columns, confirming labels and pixel data genuinely
correspond to what they claim:

![Sample digits, one per class](results/eda_sample_digits.png)

Pixel intensities are heavily bimodal (log-scale axis needed to see it) — mostly-black
backgrounds vastly outnumber bright stroke pixels — which motivates normalizing pixel
values to `[0, 1]` before training rather than using raw `[0, 255]` integers:

![Pixel intensity distribution](results/eda_pixel_distribution.png)

### 1.4 Split strategy

60% train / 20% validation / 20% test, stratified (each partition preserves the
dataset's class proportions), built in two stages: first split into train vs. a
combined val+test pool, then split that pool evenly. Both stages use the same fixed
random seed. Result: **25,200 / 8,400 / 8,400 rows.**

**Why this matters:** validation data is used *during* development to pick the best
hyperparameters — every comparison and every Optuna trial in this project used
validation accuracy, never test. Test data is touched exactly once, at the very end,
specifically so the final reported number isn't inflated by having indirectly
"peeked" at it while tuning. This discipline is what separates a rigorous evaluation
from an optimistic one.

---

## 2. Pipeline design

- **Every experiment is a config file**, loaded into Python dataclasses covering data,
  model, optimizer, and trainer settings. Changing the architecture or any
  hyperparameter is an edit to a YAML file, not a code change.
- **Images are never flattened until the last possible moment.** The data loader
  always yields images as `(1, 28, 28)` — still a 2D grid — rather than a
  pre-flattened 784-number vector. Flattening happens *inside* the MLP's `forward()`
  method. This one decision, made before the CNN was even planned, is why adding the
  CNN later required **zero changes** to data loading, cleaning, or splitting — only a
  new model class.
- **A shared base class** holds the training/validation/test-step logic and optimizer
  setup common to both architectures; the MLP and CNN classes only override how the
  network itself is built. This avoids duplicating roughly 70 lines of near-identical
  training-loop code across two model classes.
- **Two independent random seeds**: one fixed seed controls the train/val/test split,
  a separate seed controls model initialization and training randomness. Kept
  independent specifically so the cross-seed robustness check (Section 6) could vary
  one while holding the other fixed — otherwise "did the model get better" and "did
  the split get luckier" would be tangled together.
- **PyTorch Lightning** over raw PyTorch — removes training-loop and GPU-handling
  boilerplate, and provides checkpointing, early stopping, and metric-logging
  callbacks without writing them by hand. Switching between CPU (local development)
  and GPU (Kaggle) required zero code changes.

---

## 3. The MLP

### 3.1 What it does

The image is flattened into 784 numbers (one per pixel) and passed through a stack of
fully-connected layers — each computes a weighted sum of everything from the previous
layer, then applies a non-linear activation function — ending in a 10-way output, one
score per digit.

**Why this matters:** stacking linear layers with nothing non-linear between them is
mathematically pointless — the composition of any number of linear functions is still
just one linear function, so a network without activations has no more expressive
power than a single layer, regardless of depth. The activation function is what lets
the network bend its decision boundary into complex shapes instead of drawing a
straight line through the data (this is the basis of the universal approximation
theorem).

### 3.2 Baseline configuration

A deliberately plain, unbiased starting point — not hand-tuned, since the point of
running a search afterward is that the baseline only needs to be reasonable, not
optimal:

- Architecture: `784 → 256 → 128 → 64 → 10`, ReLU activation, dropout 0.25
- Optimizer: Adam, learning rate 1e-3
- Loss: plain cross-entropy (no label smoothing)
- Batch size: 64, max 30 epochs, early stopping on validation loss (patience 5)
- **Parameters: 242,762**

Trained on a Kaggle T4 GPU: early-stopped at epoch 12 of 30, reaching **validation
accuracy 0.9719**, validation loss 0.116.

### 3.3 Hyperparameter search

Testing every combination of 7+ hyperparameters by hand, or via a full grid search,
would mean tens of thousands of training runs — not realistic. This project used
**Optuna** instead: it samples the hyperparameter space and uses a model of past trial
results to bias future trials toward more promising regions, rather than searching
blindly or exhaustively.

Search space (20 real trials, run on GPU):

| Hyperparameter | Range / choices |
|---|---|
| Hidden layer shape | `[256,128]`, `[256,128,64]`, `[512,256]` |
| Activation | ReLU, GELU, tanh |
| Dropout | 0.0 – 0.5 |
| Optimizer | SGD, Adam, AdamW |
| Learning rate | 1e-4 – 1e-2 (log-uniform) |
| Weight decay | 0.0 – 1e-3 |
| Label smoothing | 0.0, 0.1 |
| Batch size | 32, 64, 128, 256 |

**Why this matters — SGD vs. Adam vs. AdamW:** SGD moves weights opposite the
gradient, scaled by the learning rate — simple, but needs careful tuning. Adam adds
momentum (smoothing updates via a running average of past gradients) and a
per-parameter adaptive learning rate (bigger steps for parameters with small,
consistent gradients; smaller steps for noisy ones) — it converges faster with much
less tuning. AdamW additionally decouples weight decay from the gradient-based update
instead of folding it into the gradient the way plain Adam does — the two are designed
to work as a pair, which is exactly what the winning configuration below converged on.

**Winner**: `hidden_dims=[512,256]`, activation **GELU**, optimizer **AdamW**,
dropout=0.2589, weight_decay=0.000435, label_smoothing=0.1, batch_size=128.
Search-reported validation accuracy: **0.9787**.

### 3.4 Retraining and final evaluation

The search itself saves no checkpoints (only hyperparameters), so the winning
configuration was retrained from scratch with checkpointing on. This run trained the
full 30 epochs without early stopping — consistent with three regularizers acting
together (dropout + weight decay + label smoothing) slowing convergence in exchange
for better generalization. The checkpoint captures the *best* epoch across training:
**validation accuracy 0.9812** — slightly higher than the search's reported 0.9787,
because the search's number is whatever validation accuracy was at a trial's *last*
epoch, while the retrain's checkpoint captures the *peak* across all 30 epochs. Not a
discrepancy in the result, a difference in what was measured. **Parameters: 535,818.**

**Why this matters — label smoothing:** normally cross-entropy loss pushes the model
toward 100% confidence in the correct class. Label smoothing softens that target (the
true class gets e.g. 0.9 instead of 1.0, spread thinly across the rest), discouraging
overconfidence and generally improving generalization and calibration. One side
effect: it keeps a loss floor even for a well-fit model, so loss values aren't
comparable between a run that uses it and one that doesn't — which is why this
config's test loss (0.5549) looks numerically worse than the baseline's (0.116) even
though its accuracy is higher. Only accuracy/precision/recall are comparable across
that difference.

**Final, one-time test evaluation: test_accuracy 0.9830, test_precision 0.9829,
test_recall 0.9830.**

### 3.5 Detailed error analysis

Full test-set confusion matrix (8,400 examples; rows = true label, columns =
predicted):

| True \ Pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Per-class acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 817 | 0 | 1 | 2 | 1 | 1 | 2 | 0 | 0 | 2 | 98.91% |
| **1** | 0 | 924 | 2 | 2 | 0 | 0 | 1 | 1 | 7 | 0 | 98.61% |
| **2** | 2 | 1 | 823 | 3 | 3 | 0 | 0 | 1 | 0 | 2 | 98.56% |
| **3** | 0 | 0 | 5 | 856 | 0 | 8 | 0 | 1 | 0 | 0 | 98.39% |
| **4** | 1 | 0 | 0 | 0 | 805 | 0 | 2 | 1 | 0 | 6 | 98.77% |
| **5** | 2 | 0 | 0 | 7 | 0 | 740 | 5 | 2 | 2 | 1 | 97.50% |
| **6** | 2 | 0 | 0 | 0 | 0 | 2 | 822 | 0 | 2 | 0 | 99.28% |
| **7** | 0 | 1 | 5 | 1 | 1 | 1 | 0 | 864 | 1 | 6 | 98.18% |
| **8** | 2 | 2 | 3 | 2 | 2 | 3 | 3 | 2 | 794 | 0 | 97.66% |
| **9** | 2 | 2 | 0 | 3 | 3 | 4 | 0 | 6 | 4 | 813 | 97.13% |

8,258 / 8,400 correct = 98.30%.

**Largest confusions**: 3→5 (8 cases) and 5→3 (7 cases) — a 3/5 confusion pair
accounting for 15 of 142 total errors (10.6%), the dominant error pattern. Both digits
share a curved-stroke structure in the upper half of the character, and a
flattened-pixel representation has no notion that these curves are spatially coherent
shapes rather than independent bright pixels. Also notable: 1→8 (7 cases), 4→9 and
9→7 (6 cases each).

**Weakest classes**: digit 9 (97.13% accuracy, 24 of 837 misclassified) and digit 5
(97.50%, 19 of 759). **Strongest**: digit 6 (99.28%).

![MLP confusion matrix](results/mlp_confusion_matrix.png)

Most misclassified examples are genuinely ambiguous even to a human eye (a looping "1"
that reads as "8", a "9" that reads as "0") — a sign these remaining errors sit close
to the ceiling of what a flattened-pixel MLP can resolve, rather than a systematic
model weakness.

![MLP misclassified examples](results/mlp_misclassified_examples.png)

---

## 4. The CNN

### 4.1 Rationale

The MLP's core weakness — no sense of spatial structure, since flattening discards
which pixels are near which other pixels — is exactly what a CNN is built to
address. A convolutional layer slides a small filter (here, 3×3 pixels) across the
image, seeing local pixel neighborhoods directly instead of treating every pixel as
an independent input. This was framed explicitly as *testing* the MLP write-up's own
"further improvement" suggestion, not adding a second model for its own sake.

Because the data pipeline already yielded un-flattened `(1, 28, 28)` images, this
required only a new model class — zero changes to data loading, cleaning, or
splitting.

### 4.2 Baseline configuration

2 convolution+pooling blocks (`conv_channels=[32,64]`, kernel 3×3, padding 1, ReLU,
`MaxPool2d(2)` after each — spatial size shrinks 28→14→7), then a fully-connected
head (`7×7×64=3136 → 128 → 10`), dropout 0.25 before the final layer. Optimizer:
Adam, lr 1e-3, batch size 64. **Parameters: 421,642** (hand-verified against the
architecture's math).

Trained on GPU: early-stopped at epoch 10 of 30, reaching **validation accuracy
0.9871** — already exceeding the MLP's fully hyperparameter-*tuned* best (0.9812),
with zero tuning of its own. This is the clearest single piece of evidence in the
project that architecture matters more than hyperparameter tuning for this problem:
the right inductive bias (convolution's "nearby pixels matter together" assumption)
outperforms extensive tuning on the wrong architecture.

### 4.3 Hyperparameter search

Search space — mirrors the MLP's search but with convolution-specific choices in
place of hidden-layer shape:

| Hyperparameter | Range / choices |
|---|---|
| Conv channels | `[16,32]` (narrow), `[32,64]` (baseline), `[64,128]` (wide) |
| Kernel size | 3, 5 |
| FC hidden size | 64, 128, 256 |
| Activation | ReLU, GELU, tanh |
| Dropout | 0.0 – 0.5 |
| Optimizer | SGD, Adam, AdamW |
| Learning rate | 1e-4 – 1e-2 (log-uniform) |
| Weight decay | 0.0 – 1e-3 |
| Label smoothing | 0.0, 0.1 |
| Batch size | 32, 64, 128, 256 |

20 trials on GPU. **Winner**: `conv_channels=[64,128]` (wide), `kernel_size=3`,
`fc_hidden_dim=256`, activation **GELU**, optimizer **Adam**, dropout=0.0150,
weight_decay=0.0000199, label_smoothing=0.1, batch_size=64. Search-reported
validation accuracy: **0.9922**.

**Why this matters:** the CNN's winning dropout and weight decay are roughly 15–20×
*smaller* than the MLP winner's (0.259 and 0.00043). Convolution is itself a strong
implicit regularizer — the same small filter is reused across every position in the
image, instead of a fully-connected layer learning a separate weight for every
pixel-to-neuron connection — far fewer independent parameters relative to
representational capacity, so the search needed much less explicit regularization to
avoid overfitting.

### 4.4 Retraining and final evaluation

Retrained with checkpointing: full 30 epochs, and this time the retrain's best-epoch
value matched the search's reported value exactly (**0.9922** both times — unlike the
MLP, this run's peak happened to land on its last epoch). **Parameters: 1,682,954**
(hand-verified: `(3×3×1×64+64) + (3×3×64×128+128) + (6272×256+256) + (256×10+10) =
1,682,954`), ~20MB checkpoint (consistent with the parameter count plus Adam's two
optimizer-state buffers per parameter).

**Final, one-time test evaluation: test_accuracy 0.9921, test_precision 0.9922,
test_recall 0.9921.**

### 4.5 Detailed error analysis

| True \ Pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Per-class acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 821 | 0 | 1 | 0 | 0 | 0 | 3 | 0 | 0 | 1 | 99.39% |
| **1** | 0 | 935 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 99.79% |
| **2** | 0 | 1 | 829 | 2 | 1 | 0 | 0 | 1 | 0 | 1 | 99.28% |
| **3** | 1 | 0 | 3 | 866 | 0 | 0 | 0 | 0 | 0 | 0 | 99.54% |
| **4** | 0 | 0 | 0 | 0 | 812 | 0 | 0 | 0 | 0 | 3 | 99.63% |
| **5** | 0 | 0 | 1 | 3 | 0 | 748 | 4 | 0 | 3 | 0 | 98.55% |
| **6** | 0 | 0 | 0 | 0 | 1 | 2 | 824 | 0 | 1 | 0 | 99.52% |
| **7** | 0 | 2 | 4 | 0 | 2 | 0 | 0 | 872 | 0 | 0 | 99.09% |
| **8** | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 805 | 3 | 99.02% |
| **9** | 1 | 0 | 1 | 0 | 2 | 2 | 0 | 6 | 2 | 823 | 98.33% |

**Largest confusion**: 9→7 (6 cases) — the single biggest off-diagonal entry in the
whole matrix. **Weakest class**: digit 9 again (98.33%), the same as the MLP.
**Strongest**: digit 1 (99.79%).

![CNN confusion matrix](results/cnn_confusion_matrix.png)

Again, the remaining misclassifications are dominated by genuinely ambiguous
handwriting rather than a systematic model weakness.

![CNN misclassified examples](results/cnn_misclassified_examples.png)

---

## 5. Comparative analysis: MLP vs. CNN

| | MLP (best) | CNN (best) |
|---|---|---|
| Parameters | 535,818 | 1,682,954 |
| Validation accuracy (balanced) | 98.12% | 99.22% |
| **Test accuracy (balanced)** | **98.30%** | **99.21%** |
| Test precision | 98.29% | 99.22% |
| Test recall | 98.30% | 99.21% |
| Test errors (of 8,400) | 142 | 66 |

Both models were evaluated on the identical stratified test split (same seed, same
8,400 rows), so this comparison is apples-to-apples. **The CNN roughly halves the
MLP's error rate** (142 → 66 errors, a 53.5% relative reduction), with a search
process equally rigorous to the MLP's.

**Where the CNN's advantage concentrates**: comparing the two confusion matrices
directly, the MLP's dominant error pattern — 3↔5 confusion (15 total cases) — is
nearly eliminated by the CNN (3→5 drops from 8 to **0**; 5→3 drops from 7 to 3). Same
story for 1→8 (7 cases → 1 case). These are exactly the errors convolution should
fix: 3/5 and 1/8 are distinguished by local stroke shape and connectivity, information
a flattened-pixel MLP has no structural access to, but which a 3×3 filter sees
directly.

**Where both models struggle equally**: 9→7 confusion sits at exactly 6 cases in
*both* matrices — untouched by the architecture change. This is a strong signal that
this particular error is closer to genuine handwriting ambiguity (a "9" with a
straight-ish descender can look like a "7") than a model limitation — a rough
approximation of where the ceiling sits for this problem.

**Convergent choices**: both winning configurations independently landed on
`activation=GELU` and `label_smoothing=0.1`, despite being searched completely
separately — weak but real evidence that these choices help on this problem somewhat
independent of architecture.

---

## 6. Robustness checks

A single accuracy number from one training run doesn't answer two important
questions: would it look this good again with a different random seed, and does it
hold up on data that isn't the pristine MNIST test set? Both were tested directly.

### 6.1 Cross-seed variance

Both winning configurations were retrained 5 times each with different random
initialization (`training_seed` 42–46), keeping the data split fixed so every retrain
is evaluated on the identical test set — isolating "how much does the number move due
to random init/shuffling alone" from split luck.

![Cross-seed variance](results/cross_seed_variance.png)

| | Mean test accuracy (balanced) | Std. dev. | Range | Original single-run result |
|---|---|---|---|---|
| MLP | 98.199% | 0.061 pp | 98.127% – 98.282% | 98.300% |
| CNN | 99.241% | 0.027 pp | 99.214% – 99.272% | 99.214% |

Both models are stable — well under a quarter of a percentage point of spread across
5 independent retrains — so the headline numbers are not lucky single runs. The CNN
is also the more *consistent* of the two (roughly half the MLP's standard deviation),
on top of being more accurate. The original MLP result sits above its own 5-seed
range; the original CNN result sits at the low end of its own range — neither
concerning (both are within ~0.1pp of their respective means), but worth stating
precisely rather than just asserting both are "consistent."

**Why this matters:** one retrain used the exact same seed as the original run. The
CNN reproduced its original result bit-for-bit; the MLP did not. This isn't a bug —
`pl.seed_everything` seeds the relevant random-number generators but doesn't force
bit-identical execution on GPU, since cuDNN's kernel scheduling has some inherent
floating-point non-determinism (the order of parallel reduction operations isn't
fixed) unless deterministic algorithms are explicitly forced, which trades away some
speed. The general phenomenon — same seed does not guarantee bit-identical GPU
results — is real and worth naming precisely rather than assumed away.

### 6.2 Distribution-shift evaluation

The already-trained models (no retraining) were evaluated on perturbed versions of
the test set — rotation, translation, and Gaussian noise — to check generalization
beyond MNIST's own narrow, curated distribution.

![Distribution-shift robustness](results/shift_robustness.png)

| Perturbation | MLP | CNN |
|---|---|---|
| Clean | 98.31% | 99.23% |
| Rotate 15° | 95.46% | 98.10% |
| Rotate 30° | 80.76% | 90.14% |
| Translate 4px | **9.42%** | **21.24%** |
| Noise σ=0.1 | 90.33% | 99.23% |
| Noise σ=0.3 | 41.68% | 78.50% |

These are plain accuracies (correct ÷ total), which is why "Clean" sits slightly above
the balanced test accuracies elsewhere in this report. The noise perturbations are
seeded (`run_shift_robustness(..., seed=0)`), so `python scripts/shift_robustness.py`
regenerates this exact table and the plot from the committed checkpoints. An earlier,
unseeded run of the same evaluation gave noise rows within 0.4 points of these.

The CNN degrades more gracefully than the MLP under every perturbation tested — most
dramatically for noise (σ=0.3: 78.50% vs. 41.68%). The translation row is the one
that needs careful interpretation: a 4-pixel shift, barely perceptible to a human,
collapses the MLP to **9.42% accuracy** — statistically indistinguishable from the
10% expected by random guessing across 10 balanced classes, meaning the model has no
usable signal left. The CNN survives somewhat better (21.24%) but is nowhere near a
usable classifier either.

**This is not evidence that CNNs are translation-invariant and MLPs aren't** — a
common misconception worth addressing directly. A plain convolution + pooling +
flatten + fully-connected CNN only gets *local, approximate* translation robustness
from the pooling operations inside its convolutional layers; the fully-connected head
at the end carries exactly the same fixed-position assumption the MLP has throughout.
Neither model was ever trained on shifted examples, so this result demonstrates a
general and well-established principle — a model does not generalize to a class of
transformation it was never trained to handle, regardless of architecture — rather
than an architecture-specific weakness or strength. It is direct empirical support for
adding data augmentation (Section 9), the standard fix for exactly this gap.

---

## 7. Engineering process

A meaningful share of the real work on this project was diagnosing environment and
tooling issues rather than modeling.

**Kaggle API authentication had changed shape.** The installed `kaggle` package
turned out to be a newer, `kagglesdk`-based rewrite, not the classic version most
tutorials describe — its `__init__.py` even eagerly tries to authenticate at import
time. The first attempt used the classic username/key pair, which was silently
*accepted* (the client only checks that both values are present, not that they're
valid) but failed with a 401 Unauthorized on the actual download call. Reading the
installed package's own source revealed the new authentication order: an access
token first, then the legacy key pair, then OAuth. Switching to the access-token
format fixed it immediately.

**PyTorch 2.6's stricter checkpoint loading broke model loading.** PyTorch 2.6
changed `torch.load`'s default from `weights_only=False` to `True`, which refuses to
unpickle arbitrary custom classes — including this project's own config dataclasses,
which had been stored directly inside checkpoints. Rather than loosening that real
security default (meant to prevent arbitrary code execution from untrusted
checkpoint files), the fix changed what gets stored: hyperparameters are now saved as
plain dictionaries, safe to unpickle under the strict default, with model
construction accepting either a dataclass or a dict.

**Kaggle Notebooks' actual input layout didn't match the documented one.** The
commonly-documented flat layout (`/kaggle/input/<slug>/`) doesn't exist in this
environment — inputs are nested one level deeper
(`/kaggle/input/competitions/<slug>/` for competitions,
`/kaggle/input/datasets/<owner>/<slug>/` for datasets). Diagnosed by printing the
actual directory tree from inside a notebook cell, since the file-browser sidebar was
showing a stale view in at least one instance. A related discovery: Kaggle
auto-extracts uploaded `.zip` resources into real folders server-side, so no
in-notebook unzip step was needed.

**The Optuna pruner is configured but inert — an honest, documented limitation.**
`optuna.pruners.MedianPruner` is passed to the study, but the training loop never
reports intermediate per-epoch metrics back to Optuna mid-training, which is what a
pruner needs to decide whether to cut a bad trial short. Every trial therefore runs to
its own early-stopping point regardless of how it's doing relative to others so far.
Left as a concrete "further improvement" rather than silently left looking more
finished than it is.

**Downloading results off Kaggle needed a workaround.** The standard Jupyter download
mechanism (`IPython.display.FileLink`) returned 404s on Kaggle's hosted environment.
Fixed by generating the download link client-side instead, as a base64-encoded
`data:` URI embedded directly in the notebook cell's output — the browser downloads
it with no server round-trip involved.

**A real bug caught before it ever ran.** While extending the search code to support
the CNN, `build_config()` was found to not be propagating the `architecture` field
into the config it constructs — meaning a CNN search trial would have silently built
an MLP-shaped config, crashing partway through what would otherwise have looked like
a normal 20-trial search. Caught during local verification, before ever running
against real data or spending a Kaggle GPU session on it.

**Kaggle's free-tier GPU session limit.** Hit the "maximum 1 interactive GPU session"
cap when starting the CNN notebook while the MLP notebook's session was still alive.
Resolved by stopping the earlier session, safe since its results were already
downloaded and verified locally.

---

## 8. Code quality and performance

A `ruff` (Python linter) audit was run across the codebase (configured rule set: `E,
W, F, B, UP, PTH, RUF`, 120-character line limit) against PEP8 and broader code-quality
rules. It auto-fixed 32 safe style modernizations (legacy `typing.List`/`Union`/
`Optional` replaced with modern `list`/`X | Y` syntax). Two issues needed a human
judgment call rather than an automatic fix: a missing explicit file encoding on a
file-read call (a genuine cross-platform risk, since this code runs on both Windows
and Kaggle's Linux containers) was fixed; a `zip()` call the linter flagged for a
`strict=True` fix was deliberately left alone, because that particular length
mismatch is intentional and handled explicitly a few lines later — applying the
suggestion there would have introduced a real bug.

Design choices worth naming: configuration as dataclasses rather than raw
dictionaries, for IDE support and typo-safety; a shared base class to avoid
duplicating training-loop code across architectures; factory functions
(`build_model`/`load_model`) instead of scattering architecture `if` checks through
the training and evaluation code; and a deliberate decision *not* to further
collapse a small amount of remaining duplication between the MLP and CNN model
classes' constructors — documented as a judgment call, since with only two
architectures the simpler, more explicit version is arguably more readable.

**A measured performance fix.** Identified while auditing the project against its
own "suitable algorithms and data structures" criterion: the data-loading
step re-read and re-cleaned the full 76MB `train.csv` from scratch on every call, even
though the result is identical every time given a fixed split seed. Across the
project this function was called roughly 54 times (baseline + 20-trial search +
retrain, for each of two architectures, plus 10 cross-seed reruns) — most of them
within the same process, making most calls fully redundant re-parses. Fix: the
load-clean-split logic was moved into a cached function, keyed on the parameters that
actually determine the result. Measured directly, not assumed: a cold call took
**4.651 seconds**; a second call from a fresh instance with the same configuration
took **effectively 0 seconds** (cache hit), with array-identical results verified via
exact equality checks, not just matching shapes.

---

## 9. Suggestions for further improvement

- **A real pruning callback for the Optuna search.** Both searches configure
  `MedianPruner`, but trials don't report intermediate metrics back to Optuna, so it
  can't currently cut a clearly-bad trial short. Wiring in per-epoch reporting would
  make a larger search (more trials, wider ranges) more efficient.
- **Data augmentation** (small rotations/shifts/translations during training) — no
  longer just a theoretical suggestion: Section 6.2 shows both models collapse under a
  4-pixel translation they were never trained to handle (MLP to chance level, CNN to
  21%). Augmentation during training is the direct fix, and the CNN should benefit
  most, since translation/rotation tolerance is exactly what convolution plus
  augmentation reinforces together.
- **Wider/longer search**, especially for the CNN — the search only varied 2-layer
  channel widths; depth (3+ convolutional blocks) wasn't explored.
- **Weight initialization / batch normalization.** Not explored here, since both
  architectures are shallow enough that vanishing/exploding gradients weren't a
  practical issue, but worth an ablation on a deeper network.
- **Transfer learning** (e.g. a small pretrained backbone, fine-tuned) — unlikely to
  help much on MNIST specifically, since training from scratch already reaches 99.2%
  and the dataset is simple enough that this is standard practice, but would be the
  natural next step on a harder, more data-hungry image dataset.
- **Adversarial robustness** — Section 6 covers natural distribution shift (rotation,
  translation, noise), not adversarially-crafted perturbations (e.g. the FGSM attack).
  A cheap, no-retraining follow-up if pursued further; full adversarial *training* was
  scoped out as disproportionate to this project.

---

## 10. Requirements checklist

| # | Requirement | Status | Notes |
|---|---|---|---|
| 1 | Download data from Kaggle | Met | Kaggle API, authenticated via a project-local access token |
| 2 | 3-way split (60/20/20) | Met | Stratified, two-stage split, exactly 25,200 / 8,400 / 8,400 rows |
| 3 | Easily customizable pipeline | Met | Every architectural/training choice is a config field loaded from YAML |
| 4 | Compare optimizers / batch sizes / activations / LRs / dropout / hidden architectures / loss functions | Met | All 7 varied across 20 real Optuna trials per architecture |
| 5 | Evaluate best model on test set | Met | Evaluated exactly once, after selection was finalized on validation data |
| 6 | GPU training | Met | Kaggle Notebooks, T4 GPU |
| 7 | Suggestions for further improvement | Met | Documented above — one suggestion (the CNN) was actually built and evaluated, not just proposed |

One nuance: for "different loss functions" specifically, plain cross-entropy was
compared against label-smoothed cross-entropy, rather than switching to an entirely
different loss family (e.g. focal loss). Label smoothing is a legitimate, genuinely
different training objective — not just a training hyperparameter — but a stricter
reading of the requirement could expect more loss-function diversity.

---

## 11. Limitations and future work

- **The Optuna pruner doesn't actually prune** — the most concrete next step, and the
  one most directly tied to search efficiency at scale.
- **CNN search only varied 2-layer channel widths**, not depth.
- **No data augmentation** — Section 6.2 turned this from a theoretical suggestion
  into a demonstrated gap: both models collapse under a 4px translation they were
  never trained to handle.
- **No weight-init or batch-normalization ablation** — not a practical issue at this
  depth, but would matter on a deeper network.
- **No transfer learning** — reasonable for MNIST specifically, but the natural
  approach on a harder, more data-hungry image task.
- **The 9→7 confusion persists in both models** — likely close to the ceiling of what
  pixel-based classification alone can resolve for genuinely ambiguous handwriting.
- **The root cause of the GPU non-determinism asymmetry in Section 6.1 wasn't
  investigated** — the CNN's same-seed retrain matched bit-for-bit, the MLP's didn't.
  Reported honestly rather than explained beyond "cuDNN scheduling isn't guaranteed
  deterministic," since a fuller answer would need kernel-level profiling.
- **Adversarial robustness untested** — natural distribution shift was measured;
  adversarially-crafted perturbations were not.

---

## 12. Conclusion

The MLP deliverable meets every stated requirement: cleaned and verified data, a
proper stratified split, a config-driven and genuinely customizable pipeline, a
systematic 8-dimension hyperparameter search rather than manual guessing, GPU
training, a single disciplined test-set evaluation (98.30% accuracy), and concrete,
evidenced suggestions for improvement. The CNN stretch goal validated the largest of
those suggestions directly: switching architecture, with an equally rigorous search
process, cut the error rate roughly in half (98.30% → 99.21% test accuracy) — and did
so specifically by resolving the MLP's structurally-explainable confusions (3↔5,
1→8) while leaving genuinely ambiguous cases (9→7) essentially untouched, exactly the
pattern expected if the improvement comes from real architectural signal rather than
noise. Two further robustness checks — cross-seed variance and distribution-shift
evaluation — extend this from "here is a number" to "here is how much to trust that
number, and where its limits are": both models are stable across retrains (under a
quarter of a percentage point of spread), but neither generalizes to transformations
it was never trained on, most starkly under small translations. That gap, not further
hyperparameter tuning, is the most consequential next step for either model.
