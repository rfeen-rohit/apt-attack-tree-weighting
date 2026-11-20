"""A numpy-only stand-in for the two scikit-learn objects this kit needs.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
scikit-learn is not installed on this machine and cannot be installed: the egress
proxy refuses PyPI, so `pip install scikit-learn` fails and there is no wheel to
vendor. Without a chunk scorer, 04_run_experiment.py cannot run at all, which means
link A1 cannot be consumed and link A2 cannot be emitted from real detections - the
anchor chain would have a hole at stage 2 for a reason that has nothing to do with
the design.

So this module trains a real classifier on the real corpus, and the flags it
produces are real detector output. It is NOT the detector the design specifies:

  * the features are byte n-grams hashed into a fixed number of buckets, computed by
    a rolling polynomial hash, where sklearn's HashingVectorizer uses a signed
    murmur3 hash over character n-grams with different collision behaviour;
  * the classifier is logistic regression trained by mini-batch gradient descent
    with L2, where the kit asks for liblinear's exact coordinate-descent solution;
  * class balancing is a scalar weight on the positive class, which is what
    class_weight="balanced" amounts to, but the optimum reached is not the same one.

Every tpr and fpr produced through this path is therefore NOT a result about
mean-or-max aggregation, and must not be compared with Kwon 2026 or with any
sklearn run. 04_run_experiment.py stamps "NUMPY STAND-IN SCORER" into the operating
point's detector field, into the results file and into the console output, and
collect_results.py prints it as not a measurement. Install scikit-learn and rerun
with --scorer sklearn to get the design's numbers.

Standard library plus numpy only.
"""
from __future__ import annotations

import numpy as np

# One 64-bit odd multiplier per n-gram length, so the three lengths land in
# different parts of the bucket space instead of colliding with each other.
_MULT = {3: 0x9E3779B97F4A7C15, 4: 0xC2B2AE3D27D4EB4F, 5: 0x165667B19E3779F9}
_BASE = np.uint64(257)


class HashedNgrams:
    """Byte n-gram hashing into `n_features` buckets, L2-normalised per row.

    Vectorised: the rolling hash for every window of a document is computed with
    numpy slicing, so no Python loop touches a byte. Rows are stored CSR-style in
    three flat arrays, which is what keeps a corpus of chunks inside memory without
    scipy.
    """

    def __init__(self, n_features=2 ** 16, ngram_range=(3, 5)):
        self.n_features = int(n_features)
        self.ngram_range = tuple(ngram_range)

    def _doc_buckets(self, text: str) -> np.ndarray:
        b = np.frombuffer(text.encode("latin-1", "replace"), dtype=np.uint8).astype(np.uint64)
        out = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(b) < n:
                continue
            h = np.zeros(len(b) - n + 1, dtype=np.uint64)
            for k in range(n):
                h = h * _BASE + b[k:len(b) - n + 1 + k]
            h = h * np.uint64(_MULT.get(n, 0x9E3779B97F4A7C15))
            out.append((h >> np.uint64(24)) % np.uint64(self.n_features))
        if not out:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(out).astype(np.int64)

    def transform(self, texts) -> "CSR":
        indptr = np.zeros(len(texts) + 1, dtype=np.int64)
        idx_parts, val_parts = [], []
        for i, t in enumerate(texts):
            buckets = self._doc_buckets(t)
            if buckets.size:
                uniq, counts = np.unique(buckets, return_counts=True)
                v = counts.astype(np.float32)
                nrm = np.sqrt(float((v * v).sum())) or 1.0
                v /= nrm
            else:
                uniq = np.zeros(0, dtype=np.int64)
                v = np.zeros(0, dtype=np.float32)
            idx_parts.append(uniq)
            val_parts.append(v)
            indptr[i + 1] = indptr[i] + uniq.size
        indices = (np.concatenate(idx_parts) if idx_parts
                   else np.zeros(0, dtype=np.int64))
        values = (np.concatenate(val_parts) if val_parts
                  else np.zeros(0, dtype=np.float32))
        return CSR(indices, values, indptr, self.n_features)


class CSR:
    """The smallest sparse matrix that supports what the trainer needs."""

    def __init__(self, indices, values, indptr, n_features):
        self.indices = indices
        self.values = values
        self.indptr = indptr
        self.shape = (len(indptr) - 1, int(n_features))

    def __len__(self):
        return self.shape[0]

    def rows(self, sel):
        """A new CSR holding only the rows in `sel` (an index array)."""
        sel = np.asarray(sel, dtype=np.int64)
        lens = self.indptr[sel + 1] - self.indptr[sel]
        indptr = np.zeros(len(sel) + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
        take = np.concatenate([np.arange(self.indptr[s], self.indptr[s + 1])
                               for s in sel]) if len(sel) else np.zeros(0, dtype=np.int64)
        return CSR(self.indices[take], self.values[take], indptr, self.shape[1])

    def dot(self, w) -> np.ndarray:
        """X @ w, row by row, without materialising anything dense per row."""
        n = self.shape[0]
        if self.indices.size == 0:
            return np.zeros(n, dtype=np.float64)
        prod = self.values.astype(np.float64) * w[self.indices]
        # segment sums over the CSR row boundaries
        cs = np.concatenate([[0.0], np.cumsum(prod)])
        return cs[self.indptr[1:]] - cs[self.indptr[:-1]]

    def rmatvec_weighted(self, r) -> np.ndarray:
        """X.T @ r, as a bucket histogram weighted by the per-row residual."""
        if self.indices.size == 0:
            return np.zeros(self.shape[1], dtype=np.float64)
        counts = self.indptr[1:] - self.indptr[:-1]
        per_nz = np.repeat(r.astype(np.float64), counts)
        return np.bincount(self.indices, weights=self.values.astype(np.float64) * per_nz,
                           minlength=self.shape[1])


class LogisticSGD:
    """Logistic regression by mini-batch gradient descent, L2 penalised.

    Not liblinear. The optimum it reaches is close but not identical, which is
    exactly why every number downstream of it is labelled a stand-in.
    """

    def __init__(self, epochs=12, batch=4096, lr=1.0, l2=1e-6, balanced=True,
                 random_state=0):
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.balanced = bool(balanced)
        self.random_state = int(random_state)
        self.w = None
        self.b = 0.0

    def fit(self, X: CSR, y):
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0
        npos, nneg = float((y == 1).sum()), float((y != 1).sum())
        if self.balanced and npos and nneg:
            wpos, wneg = (npos + nneg) / (2 * npos), (npos + nneg) / (2 * nneg)
        else:
            wpos = wneg = 1.0
        sw = np.where(y == 1, wpos, wneg)
        rng = np.random.default_rng(self.random_state)
        for ep in range(self.epochs):
            order = rng.permutation(n)
            lr = self.lr / (1.0 + ep)
            for s in range(0, n, self.batch):
                sel = order[s:s + self.batch]
                Xb = X.rows(sel)
                z = Xb.dot(self.w) + self.b
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                r = (p - y[sel]) * sw[sel]
                g = Xb.rmatvec_weighted(r) / len(sel) + self.l2 * self.w
                self.w -= lr * g
                self.b -= lr * float(r.mean())
        return self

    def predict_proba(self, X: CSR) -> np.ndarray:
        z = X.dot(self.w) + self.b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1.0 - p, p])


DESCRIPTION = ("NUMPY STAND-IN SCORER (byte n-gram hashing + mini-batch logistic "
               "regression). scikit-learn is not installable on this machine, so this "
               "is not HashingVectorizer + liblinear and its rates are NOT a result "
               "about aggregation.")


def _selftest():
    rng = np.random.default_rng(0)
    vec = HashedNgrams(n_features=2 ** 12, ngram_range=(3, 5))
    bad = ["<?php eval($_POST['x']); system($_GET['c']); ?>" + "a" * i for i in range(60)]
    good = ["function render($tpl){ return view($tpl); } // layout " + "b" * i
            for i in range(60)]
    texts = bad + good
    y = np.array([1] * len(bad) + [0] * len(good))
    X = vec.transform(texts)
    assert X.shape[0] == len(texts)
    # dot and rmatvec agree with a dense reference on a small case
    dense = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.indptr[i], X.indptr[i + 1]):
            dense[i, X.indices[j]] += X.values[j]
    w = rng.normal(size=X.shape[1])
    assert np.allclose(X.dot(w), dense @ w, atol=1e-6)
    r = rng.normal(size=X.shape[0])
    assert np.allclose(X.rmatvec_weighted(r), dense.T @ r, atol=1e-6)
    sub = X.rows(np.array([3, 7, 11]))
    assert np.allclose(sub.dot(w), (dense @ w)[[3, 7, 11]], atol=1e-6)

    m = LogisticSGD(epochs=20, batch=32, random_state=1).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert p[y == 1].mean() > p[y == 0].mean(), "it must at least separate the training set"
    print("numpy_scorer.py: all self-tests passed")


if __name__ == "__main__":
    _selftest()
