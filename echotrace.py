#!/usr/bin/env python3
"""
EchoTrace (command-line version) - finds the same patient on both sides of a
train/test split in a medical imaging dataset, even when patient IDs have been
deleted, and writes a fair split that keeps each group on one side.

Same method as the browser tool (index.html):
  * every image -> a 32x32 greyscale vector (mean-centred, unit length) and a 64-bit DCT hash
  * an automatic cut-off learned from the folder itself: a robust normal fit to the
    Fisher-z similarity of random pairs; flag only matches an unrelated image would
    reach by chance with probability ALPHA across all n-1 comparisons
  * near-copies joined into groups (union-find); a fair split keeps every group on one side
  * memorisation score: accuracy of copying the label of the most similar training image

Usage:
  python echotrace.py PATH_TO_FOLDER [--out results] [--mask] [--alpha 0.01]

Folder layout (all optional):
  train/ and test/ (or val/) folders anywhere in the path -> checks that split
  a folder per class inside them (train/benign/...)        -> memorisation score
  file names like PATIENT_001.png or a folder per patient  -> precision and recall

Needs Python 3.9+, numpy and Pillow.  Licence: MIT.
"""
import argparse, csv, json, os, re, sys, time
from statistics import NormalDist
import numpy as np
from PIL import Image, ImageFilter

VERSION = "2.3"
FLOOR = 0.80  # never flag below this similarity; checked against datasets with known patient IDs
STRICT = 0.98  # near-identical overall pattern
DETAIL = 0.70  # fine-detail check: a verified copy must also match in edges and texture (128x128, high-pass).
               # Calibration: 916 knee X-ray pairs from different patients that look alike overall all score <= 0.61;
               # re-filtered copies (Br35H) and same-patient brain MRI matches score >= 0.87 in 95% of cases.
TOPK = 5       # training candidates checked per test image
SHORTCUT_MARGIN = 0.10  # flag a shortcut when file facts alone beat chance by 10 points of balanced accuracy
SIZE = 32
IMG_RE = re.compile(r"\.(png|jpe?g|bmp|gif|webp|tiff?)$", re.I)
SKIP_RE = re.compile(r"(_gt|_mask|mask_|annotation)", re.I)
TRAIN_RE = re.compile(r"^(train|training|trainset|train_set|train-set)$", re.I)
TEST_RE = re.compile(r"^(test|testing|testset|test_set|test-set|val|valid|validation)$", re.I)
SPLITNAME_RE = re.compile(r"^(train|training|trainset|train_set|train-set|test|testing|testset|test_set|test-set|val|valid|validation)$", re.I)
HASH_LEVELS = [0, 4, 8, 12]

# DCT-II basis for the 64-bit hash
_K = np.arange(SIZE)
BASIS = np.cos(np.pi * _K[:, None] * (2 * _K[None, :] + 1) / (2 * SIZE))


def list_images(root):
    files = []
    for d, _, names in os.walk(root):
        for f in names:
            if IMG_RE.search(f) and not SKIP_RE.search(f):
                files.append(os.path.relpath(os.path.join(d, f), root))
    return sorted(files, key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p)])


def split_of(rel):
    for seg in rel.replace("\\", "/").split("/")[:-1]:
        if TRAIN_RE.match(seg): return "train"
        if TEST_RE.match(seg): return "test"
    return None


def detect_subjects(rels):
    def plausible(a):
        if any(x is None for x in a): return False
        c = {}
        for x in a: c[x] = c.get(x, 0) + 1
        if len(c) < 10 or len(c) >= len(a): return False
        sizes = sorted(c.values())
        return sizes[len(sizes) // 2] <= 60
    prefix = [(os.path.splitext(os.path.basename(r))[0].split("_")[0] if "_" in os.path.basename(r) else None) for r in rels]
    folder = []
    for r in rels:
        p = r.replace("\\", "/").split("/")
        last = p[-2] if len(p) > 1 else None
        folder.append(last if last and not re.match(r"^(train|training|test|testing|val|valid|validation|images?|data)$", last, re.I) else None)
    if plausible(prefix): return prefix, "file-name prefix"
    if plausible(folder): return folder, "patient folders"
    return None, None


def detect_labels(rels, how):
    if how == "patient folders": return None
    lab = []
    for r in rels:
        p = r.replace("\\", "/").split("/")
        last = p[-2] if len(p) > 1 else None
        lab.append(last if last and not SPLITNAME_RE.match(last) else None)
    if any(x is None for x in lab): return None
    return lab if 2 <= len(set(lab)) <= 30 else None


def features(path, mask):
    im = Image.open(path).convert("L")
    if mask:
        w, h = im.size
        im = im.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.85)))
    g = np.asarray(im.resize((SIZE, SIZE), Image.LANCZOS), np.float64)
    v = g.ravel() - g.mean()
    v /= (np.linalg.norm(v) or 1.0)
    c = (BASIS @ g @ BASIS.T)[:8, :8].ravel()
    rest = np.sort(c[1:])
    med = (rest[31] + rest[32]) / 2
    bits = (c > med).astype(np.uint64)
    h = np.uint64(0)
    for b in bits: h = (h << np.uint64(1)) | b
    return v.astype(np.float32), h


def detail_vec(path, mask, N=128, sigma=3):
    """Fine detail only: the image minus a blurred copy of itself, so overall brightness and layout don't count."""
    im = Image.open(path).convert("L")
    if mask:
        w, h = im.size
        im = im.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.85)))
    im = im.resize((N, N), Image.LANCZOS)
    a = np.asarray(im, np.float32); b = np.asarray(im.filter(ImageFilter.GaussianBlur(sigma)), np.float32)
    v = (a - b).ravel(); v -= v.mean()
    return v / (np.linalg.norm(v) or 1.0)


class DetailCache:
    def __init__(self, root, mask, cap=3000):
        self.root, self.mask, self.cap, self.d = root, mask, cap, {}
    def get(self, rel):
        v = self.d.get(rel)
        if v is None:
            if len(self.d) >= self.cap: self.d.pop(next(iter(self.d)))
            v = self.d[rel] = detail_vec(os.path.join(self.root, rel), self.mask)
        return v
    def sim(self, a, b): return float(self.get(a) @ self.get(b))


def learn_cutoff(V, alpha, sample=200000, seed=20261101):
    n = len(V)
    rng = np.random.default_rng(seed)
    m = int(min(sample, n * (n - 1) // 2))
    i = rng.integers(0, n, m); j = rng.integers(0, n, m)
    j = np.where(j == i, (j + 1) % n, j)
    r = np.einsum("ij,ij->i", V[i], V[j])
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    mu = np.median(z); sd = max(1.4826 * np.median(np.abs(z - mu)), 1e-6)
    cut = float(np.tanh(mu + sd * NormalDist().inv_cdf(1 - alpha / max(1, n - 1))))
    return min(cut, 0.9999), float(np.tanh(mu))


def file_facts(path):
    """Facts about the file with no picture content: size, shape, colour mode, format."""
    with Image.open(path) as im:
        w, h = im.size; mode = im.mode
    ext = os.path.splitext(path)[1].lower()
    return [np.log(w), np.log(h), np.log(w / h), float(mode == "L"), float(mode == "RGB"), float(mode not in ("L", "RGB")),
            float(ext in (".jpg", ".jpeg")), float(ext == ".png"), float(ext not in (".jpg", ".jpeg", ".png"))]


def shortcut_score(X, labels, key, folds=5, k=5, seed=20261108):
    """Shortcut check: can the label be predicted from file facts alone?
    k-nearest-neighbour vote, cross-validated with every group kept inside one fold (so copies can't help).
    Returns balanced accuracy and chance level."""
    lab = np.array(labels, dtype=object); X = np.asarray(X, float); n = len(lab)
    sd = X.std(0); sd[sd == 0] = 1.0; Z = (X - X.mean(0)) / sd
    groups = sorted(set(str(g) for g in key)); rng = np.random.default_rng(seed); rng.shuffle(groups)
    fold_of = {g: i % folds for i, g in enumerate(groups)}; f = np.array([fold_of[str(g)] for g in key])
    pred = np.empty(n, dtype=object); sq = (Z ** 2).sum(1)
    for q in range(folds):
        te = np.nonzero(f == q)[0]; tr = np.nonzero(f != q)[0]
        if len(te) == 0 or len(tr) < k: continue
        for s0 in range(0, len(te), 256):
            b = te[s0:s0 + 256]
            D = sq[b][:, None] + sq[tr][None, :] - 2 * Z[b] @ Z[tr].T
            D += rng.random(D.shape) * 1e-9          # break exact ties at random
            nn = np.argpartition(D, k, axis=1)[:, :k]
            for row, idx in zip(b, nn):
                vals, cnt = np.unique(lab[tr[idx]].astype(str), return_counts=True); pred[row] = vals[cnt.argmax()]
    classes = sorted(set(lab.astype(str)))
    rec = [np.mean(pred[lab.astype(str) == c].astype(str) == c) for c in classes if np.any(lab.astype(str) == c)]
    return float(np.mean(rec)), 1.0 / len(classes)


def fair_split(key, labels, frac, seed=20261107):
    groups = {}
    for k, g in enumerate(key): groups.setdefault(g, []).append(k)
    lab = labels or ["all"] * len(key)
    total = {}
    for l in lab: total[l] = total.get(l, 0) + 1
    target = {l: v * frac for l, v in total.items()}; in_test = {l: 0 for l in total}
    order = list(groups.values()); np.random.default_rng(seed).shuffle(order)
    out = [None] * len(key)
    for members in order:
        cnt = {}
        for k in members: cnt[lab[k]] = cnt.get(lab[k], 0) + 1
        maj = max(cnt, key=cnt.get)
        go = in_test[maj] + len(members) / 2 <= target[maj]
        for k in members:
            out[k] = "test" if go else "train"
            if go: in_test[lab[k]] += 1
    return out, len(groups)


def main():
    ap = argparse.ArgumentParser(description="EchoTrace: find hidden patient overlap in a medical imaging dataset.")
    ap.add_argument("folder"); ap.add_argument("--out", default="echotrace_results")
    ap.add_argument("--mask", action="store_true", help="ignore borders and on-screen text (central crop)")
    ap.add_argument("--alpha", type=float, default=0.01, help="chance a non-copy image is flagged (default 0.01)")
    ap.add_argument("--floor", type=float, default=FLOOR, help="never flag below this similarity (default 0.80; set 0 to turn off)")
    a = ap.parse_args()
    t0 = time.time()
    rels = list_images(a.folder)
    if len(rels) < 2: sys.exit("Fewer than two images found.")
    V, H, kept = [], [], []
    for k, r in enumerate(rels):
        try:
            v, h = features(os.path.join(a.folder, r), a.mask); V.append(v); H.append(h); kept.append(r)
        except Exception:
            pass
        if (k + 1) % 1000 == 0: print(f"  read {k + 1}/{len(rels)} images", file=sys.stderr)
    V = np.stack(V); H = np.array(H, dtype=np.uint64); n = len(kept)
    cut_learned, typical = learn_cutoff(V, a.alpha)
    cut = max(cut_learned, a.floor)
    split = [split_of(r) for r in kept]
    tr = np.array([s == "train" for s in split]); te = np.array([s == "test" for s in split])
    has_split = tr.any() and te.any()
    subj, how = detect_subjects(kept)
    labels = detect_labels(kept, how)

    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    best = np.full(n, -2.0, np.float32); best_tr = np.full(n, -2.0, np.float32); best_tr_j = np.full(n, -1)
    cand = {}  # test image -> its TOPK training candidates at or above the cut-off
    hash_cross = {t: np.zeros(n, bool) for t in HASH_LEVELS}
    pairs = []
    B = 1024
    for s0 in range(0, n, B):
        S = V[s0:s0 + B] @ V.T
        for ii in range(S.shape[0]):
            i = s0 + ii; row = S[ii]; row[i] = -2
            best[i] = row.max()
            js = np.nonzero(row[i + 1:] >= cut)[0] + i + 1
            for j in js:
                ra, rb = find(i), find(j)
                if ra != rb: parent[ra] = rb
                pairs.append((i, int(j), float(row[j])))
            if has_split and te[i]:
                rt = np.where(tr, row, -2); j = int(rt.argmax()); best_tr[i] = rt[j]; best_tr_j[i] = j
                if rt[j] >= cut:
                    top = np.argpartition(-rt, TOPK)[:TOPK]
                    cand[i] = [int(k) for k in top[np.argsort(-rt[top])] if rt[k] >= cut]
                d = np.bitwise_count(np.bitwise_xor(H[tr], H[i])) if tr.any() else np.array([99])
                for t in HASH_LEVELS: hash_cross[t][i] = d.min() <= t
        print(f"  compared {min(s0 + B, n)}/{n}", file=sys.stderr)
    root = np.array([find(k) for k in range(n)])
    sizes = {}
    for r in root: sizes[r] = sizes.get(r, 0) + 1
    n_groups = sum(1 for v in sizes.values() if v > 1); largest = max(sizes.values())
    mixed = 0
    if labels:
        gl = {}
        for k, r in enumerate(root): gl.setdefault(r, set()).add(labels[k])
        mixed = sum(1 for r, s in gl.items() if sizes[r] > 1 and len(s) > 1)
    key = subj if subj else root.tolist()

    summary = dict(version=VERSION, folder=os.path.abspath(a.folder), images=n, masked=a.mask, alpha=a.alpha,
                   cutoff=round(cut, 4), cutoff_learned=round(cut_learned, 4), floor=a.floor,
                   floor_applied=bool(cut > cut_learned), typical_unrelated_r=round(typical, 3), near_copy_pairs=len(pairs),
                   near_copy_groups=n_groups, largest_group=largest, groups_mixing_labels=(mixed if labels else None),
                   patients_from=how, labels_found=bool(labels))
    if has_split:
        n_te = int(te.sum()); hit = int(((best_tr >= cut) & te).sum())
        has_train_root = set(root[tr]); chain = int(sum(1 for k in range(n) if te[k] and root[k] in has_train_root))
        strict = int(((best_tr >= STRICT) & te).sum())
        dc = DetailCache(a.folder, a.mask); verified_pairs = []
        for i, js in cand.items():
            for j in js:
                ds = dc.sim(kept[i], kept[j])
                if ds >= DETAIL:
                    verified_pairs.append((i, j, ds)); break
        verified = len(verified_pairs)
        summary_verified = dict(test_with_verified_copy_in_train=verified,
                                test_with_verified_copy_pct=round(verified / int(te.sum()) * 100, 1))
        summary.update(train=int(tr.sum()), test=n_te, test_with_near_copy_in_train=hit,
                       test_with_near_copy_pct=round(hit / n_te * 100, 1), expected_false_flags=round(a.alpha * n_te, 1),
                       test_with_strict_copy_in_train=strict, test_with_strict_copy_pct=round(strict / n_te * 100, 1),
                       **summary_verified,
                       test_connected_by_chains=chain,
                       hash_cross_split_test_images={str(t): int(hash_cross[t].sum()) for t in HASH_LEVELS})
        if labels:
            ti = np.nonzero(te)[0]; trn = np.nonzero(tr)[0]
            lab = np.array(labels, dtype=object); keyarr = np.array([str(x) for x in key], dtype=object)
            copy = np.mean([lab[best_tr_j[i]] == lab[i] for i in ti])
            ok = 0
            for s0 in range(0, len(ti), 512):
                blk = ti[s0:s0 + 512]; S = V[blk] @ V[trn].T
                S[keyarr[blk][:, None] == keyarr[trn][None, :]] = -2
                ok += int(np.sum(lab[trn[S.argmax(1)]] == lab[blk]))
            vals, cnts = np.unique(lab[ti].astype(str), return_counts=True)
            summary.update(memorisation_copy_nearest_pct=round(float(copy) * 100, 1),
                           memorisation_other_groups_only_pct=round(ok / len(ti) * 100, 1),
                           majority_class_pct=round(cnts.max() / len(ti) * 100, 1))
    if subj:
        tp = sum(1 for i, j, _ in pairs if subj[i] == subj[j])
        cnt = {}
        for s in subj: cnt[s] = cnt.get(s, 0) + 1
        same_pairs = sum(c * (c - 1) // 2 for c in cnt.values())
        summary.update(precision=round(tp / len(pairs), 3) if pairs else None, recall=round(tp / same_pairs, 3) if same_pairs else None)
    if labels and len(set(labels)) > 1:
        facts = [file_facts(os.path.join(a.folder, r)) for r in kept]
        ba, chance = shortcut_score(facts, labels, key)
        summary.update(shortcut_file_facts_balanced_accuracy=round(ba, 3), shortcut_chance=round(chance, 3),
                       shortcut_flag=bool(ba - chance >= SHORTCUT_MARGIN))
    frac = float(te.sum() / (tr.sum() + te.sum())) if has_split else 0.2
    fair, ngr = fair_split(key, labels, frac)
    summary.update(fair_split_train=fair.count("train"), fair_split_test=fair.count("test"), fair_split_groups=ngr,
                   seconds=round(time.time() - t0, 1))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f: json.dump(summary, f, indent=1)
    with open(os.path.join(a.out, "near_copy_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_a", "image_b", "similarity", "split_a", "split_b"])
        for i, j, r in sorted(pairs, key=lambda p: -p[2]): w.writerow([kept[i], kept[j], f"{r:.4f}", split[i] or "", split[j] or ""])
    if has_split:
        with open(os.path.join(a.out, "verified_copies.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["test_image", "training_image", "detail_similarity"])
            for i, j, ds in verified_pairs: w.writerow([kept[i], kept[j], f"{ds:.4f}"])
    gname = {}
    with open(os.path.join(a.out, "fair_split.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "original_split", "label", "group", "fair_split"])
        for k, r in enumerate(kept):
            g = key[k]; gname.setdefault(g, str(g) if subj else f"G{len(gname) + 1:05d}")
            w.writerow([r, split[k] or "", labels[k] if labels else "", gname[g], fair[k]])
    lines = [f"# EchoTrace report", "", f"Folder: `{summary['folder']}`  ", f"Images: {n} · cut-off r ≥ {cut:.3f}{' (raised to the ' + format(a.floor, '.2f') + ' floor)' if cut > cut_learned else ''} · borders and text {'ignored' if a.mask else 'included'} · EchoTrace {VERSION}", ""]
    if has_split:
        lines += [f"**{summary['test_with_near_copy_in_train']} of {summary['test']} test images ({summary['test_with_near_copy_pct']}%) have a near-copy in training.** By chance alone, about {summary['expected_false_flags']} would be expected.",
                  f"**Verified copies: {verified} test images ({summary['test_with_verified_copy_pct']}%)** match a training image overall *and* in fine detail (≥ {DETAIL}). This is the safe number to quote: on uniformly cropped scans (e.g. knee X-rays) different patients can look alike overall, and the fine-detail check removes them.",
                  f"Through chains of near-copies, {summary['test_connected_by_chains']} test images connect to training.", ""]
        if labels:
            lines += [f"Memorisation score: copying the nearest training image scores **{summary['memorisation_copy_nearest_pct']}%**; never copying from the same group scores {summary['memorisation_other_groups_only_pct']}% (majority class {summary['majority_class_pct']}%).", ""]
    if "shortcut_flag" in summary:
        sb, sc = summary["shortcut_file_facts_balanced_accuracy"], summary["shortcut_chance"]
        lines += [(f"**Shortcut warning:** the label can be predicted from file facts alone (image size, shape, colour mode, format), with no picture content: balanced accuracy **{sb * 100:.0f}%** against {sc * 100:.0f}% chance. A model can score well here without looking at the anatomy."
                   if summary["shortcut_flag"] else
                   f"Shortcut check: file facts alone (image size, shape, colour mode, format) predict the label at {sb * 100:.0f}% balanced accuracy, against {sc * 100:.0f}% chance. No shortcut found."), ""]
    lines += [f"Near-copy groups: {n_groups} (largest {largest})" + (f"; groups mixing labels: {mixed}" if labels else "") + ".",
              f"Fair split written: {summary['fair_split_train']} train / {summary['fair_split_test']} test, no group on both sides.", "",
              "Limits: groups come from image similarity. Different-looking images of one patient can be missed, so overlap counts are lower bounds."]
    with open(os.path.join(a.out, "report.md"), "w") as f: f.write("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
