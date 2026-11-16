<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/echotrace-logo-dark.svg">
  <img src="assets/echotrace-logo.svg" alt="EchoTrace" width="360">
</picture>

# EchoTrace

**Clarity before trust.** The same patient can "echo" across the training and test sets of a medical imaging dataset. Then a model is tested on people it has already seen, and its score measures memory, not diagnosis. Public datasets delete patient IDs for privacy, so nobody can check. EchoTrace finds those patients from the images themselves, and writes a fair split that keeps each one on one side.

Made by Arittra Ghosh, Bunclody Community College, Ireland, for the Stripe Young Scientist & Technology Exhibition 2027.

## Two ways to use it

**1. In a browser (no installation).** Open `index.html` and choose a folder. Everything runs on your own computer; no image is uploaded anywhere.

**2. From the command line (for researchers and large datasets).**

```
pip install -r requirements.txt
python echotrace.py path/to/dataset --out results
python echotrace.py path/to/dataset --out results --mask    # ignore borders and on-screen text
```

It writes these files to `results/`:

| File | What it holds |
|---|---|
| `report.md` | A short readable summary |
| `summary.json` | Every number, for papers and scripts |
| `near_copy_pairs.csv` | Every flagged pair, closest first |
| `verified_copies.csv` | Test images whose training match passed the fine-detail check (when there is a split) |
| `fair_split.csv` | A new train/test split with every group on one side |

## Folder layout

- `train/` and `test/` (or `val/`) folders anywhere in the path: EchoTrace checks that split.
- A folder per class inside them (`train/benign/…`): it also reports the **memorisation score**, the accuracy of simply copying the label of the most similar training image.
- File names like `PATIENT001_xx.png`, or a folder per patient: it also reports precision and recall against those patients.

## How it works

1. Each image becomes a 32 × 32 greyscale pattern and a 64-bit image hash.
2. EchoTrace learns what *unrelated* images look like in that dataset, from 200,000 random pairs, and flags only matches that chance can't explain (by default, a 1% chance per image across all its comparisons). It reports how many false flags chance alone would produce. Since v2.1 the cut-off never drops below a similarity floor of 0.80, set by checking datasets where the true patient is known (`--floor` changes it).
3. **Fine-detail check (v2.2).** For each flagged test image, its closest training images are compared again at 128 × 128 using only fine detail (edges and texture, with overall brightness and layout removed). A match counts as a **verified copy** only if it also scores at least 0.70 here. This matters on uniformly cropped scans: on knee X-rays, different people's knees match at r = 0.99 overall, but all 916 such pairs score 0.61 or less in fine detail, while re-filtered copies of one scan score 0.87 or more in 95% of cases.
4. Near-copies are joined into groups. It warns if a group mixes labels, which suggests a false link.
5. **Shortcut check (v2.3).** When there are class folders, EchoTrace tries to predict each image's label from file facts alone: width, height, shape, colour mode and file format, with no picture content. It uses a 5-nearest-neighbour vote in 5 folds, keeping near-copy groups (or patients) together so copies can't help, and reports balanced accuracy next to chance. It warns when the score is at least 10 points above chance: a model could then score well by learning how each class was saved, not the anatomy. (The browser version uses size, shape and format, since it can't read the colour mode.)
6. The fair split keeps every group on one side and keeps each label's share of the test set.
7. **Zip files and automatic split (v2.4, browser).** The web page opens a .zip directly or takes a folder or zip dragged onto it; images are read one at a time from the zip, never all unpacked. If a folder has no train/test folders, the page splits it the usual way (80/20 at random, image by image, each label keeping its share, fixed seed) and reports how many test images would have a verified copy in training. Check: on BUSI with borders ignored this gives 9.3%, against 9.0% averaged over 200 random splits in the earlier analysis.

## Tested

| Check | Result |
|---|---|
| BUSI breast ultrasound vs a curated duplicate list | precision 0.975, recall 0.952 (8 bits) |
| BUS-UCLM breast ultrasound vs patient codes | pair precision 0.91 at the 0.80 floor (chance 0.031) |
| Kaggle brain MRI, people rebuilt from slices | exactly 200 groups of 32, no group mixing labels |
| TN3K thyroid (patient-separated by its authors) | 0 verified copies: clean |
| Knee X-ray (OAI; patient numbers in file names, patient-separated) | 0 verified copies: clean. Without the fine-detail check, 117 false alarms |
| Independent paper on the same brain MRI datasets (arXiv:2610.00421) | same answers: 88 vs 88 exact copies; 83.3% vs 82.9% overlap between datasets; 502 vs 460 near-twins |
| Shortcut check (v2.3) | flagged a Kaggle PCOS ultrasound set (100%, chance 50%: the two classes never share an image size), Nickparvar brain MRI (51%, chance 25%) and SARTAJ brain MRI (44%, chance 25%); no flag on knee X-ray (20%, chance 20%), skin (51%, chance 50%) or Alzheimer MRI (25%, chance 25%). Browser and command line agree within 1 point |
| Speed (command line) | 6,400 images in about 8 seconds; 11,519 in about 17 seconds |

The full list of datasets checked so far will be added here once each dataset's authors have been contacted.

## Limits

EchoTrace finds images that look almost the same, such as copies, re-filtered copies and neighbouring frames or slices from one scan. It can miss the same patient scanned on a different day, so its verified counts are **lower bounds**, and its fair split is a minimum fix. On uniformly cropped images (knee X-rays, standard heart views) the overall-match count can include different patients; quote the verified count. When real patient IDs exist, use them.

## The real fix

Dataset makers should release an **anonymised patient number** with every image: random, specific to the dataset, with no key kept (otherwise it is pseudonymised rather than anonymous under GDPR), plus a session number and the official split file. FETAL PLANES already does this. EchoTrace is for the many datasets already released without one.

## Cite and licence

Code: MIT licence. If you use EchoTrace, please cite it using `CITATION.cff`.

Questions, corrections or a dataset to add: open an issue.

The code was written with an AI assistant (Claude). The design, the choice of tests, the preregistrations (osf.io/ge5b2, osf.io/8f6bv) and the blind checks on new datasets are the author's.
