"""
HMM-based behavior motif discovery from per-frame SSL representations.

Input:
    data: np.ndarray of shape (num_sequence, L, D)
          num_sequence videos/sessions, each with L frames, D-dim features.
          Assumes all sequences share the same L (no padding). If your
          sequences have variable length and you've zero-padded to a common
          L, see the `seq_lengths` note in `subsample_sequences`.

Pipeline:
    1. Subsample each sequence independently (stride-based) to preserve
       temporal order and avoid cross-sequence contamination.
    2. Flatten to (N, D) + lengths array for hmmlearn.
    3. Standardize + PCA.
    4. Fit GaussianHMM with several n_components, pick by held-out
       (sequence-level) log-likelihood.
    5. Inspect dwell times / transition matrix.
    6. Merge redundant states via hierarchical clustering on state means.
    7. Temporally smooth the decoded state sequence.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import median_filter
from hmmlearn import hmm


# ---------------------------------------------------------------------------
# 1. Subsample each sequence independently
# ---------------------------------------------------------------------------
def subsample_sequences(data, stride=2, seq_lengths=None):
    """
    data: (num_sequence, L, D) array.
    stride: keep every `stride`-th frame within each sequence.
    seq_lengths: optional list/array of true lengths per sequence, if
                 sequences were zero-padded to a common L. If provided,
                 only the first seq_lengths[i] frames of sequence i are used.

    Returns:
        X: (N_sub, D) concatenated subsampled features
        lengths: list of ints, per-sequence subsampled frame counts
                 (sums to N_sub) -- pass directly to hmmlearn's fit/predict
    """
    num_seq, L, D = data.shape
    feats_list = []
    lengths = []

    for i in range(num_seq):
        true_len = seq_lengths[i] if seq_lengths is not None else L
        seq = data[i, :true_len]          # (true_len, D)
        sub = seq[::stride]               # (ceil(true_len/stride), D)
        feats_list.append(sub)
        lengths.append(len(sub))

    X = np.concatenate(feats_list, axis=0)
    assert sum(lengths) == len(X)
    return X, lengths


# ---------------------------------------------------------------------------
# 2. Preprocess: standardize + PCA (no L2 normalization for HMM -- Gaussian
#    emissions handle scale via covariance, and normalization distorts it)
# ---------------------------------------------------------------------------
def preprocess(X, n_pca=15, random_state=42):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_std)
    print(f"PCA explained variance (top {n_pca}): "
          f"{pca.explained_variance_ratio_.sum():.3f}")
    return X_pca, scaler, pca


# ---------------------------------------------------------------------------
# 3. Split sequences into train/val for model selection (by sequence, not
#    frame, to avoid leakage)
# ---------------------------------------------------------------------------
def split_by_sequence(X, lengths, val_frac=0.15, random_state=42):
    rng = np.random.RandomState(random_state)
    n_seq = len(lengths)
    idx = rng.permutation(n_seq)
    n_val = max(1, int(n_seq * val_frac))
    val_idx = set(idx[:n_val])

    offsets = np.cumsum([0] + lengths)
    train_feats, val_feats = [], []
    train_lengths, val_lengths = [], []

    for i, L in enumerate(lengths):
        seg = X[offsets[i]:offsets[i + 1]]
        if i in val_idx:
            val_feats.append(seg)
            val_lengths.append(L)
        else:
            train_feats.append(seg)
            train_lengths.append(L)

    X_train = np.concatenate(train_feats, axis=0)
    X_val = np.concatenate(val_feats, axis=0)
    return X_train, train_lengths, X_val, val_lengths


# ---------------------------------------------------------------------------
# 4. Fit HMMs across a range of n_components, select by val log-likelihood
# ---------------------------------------------------------------------------
def select_n_components(X_train, train_lengths, X_val, val_lengths,
                         candidates=(10, 20, 30, 50), n_iter=200,
                         covariance_type="diag", random_state=42, n_restarts=3):
    results = {}
    for n_comp in candidates:
        best_ll = -np.inf
        best_model = None
        for r in range(n_restarts):
            model = hmm.GaussianHMM(
                n_components=n_comp,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=random_state + r,
                verbose=False,
            )
            try:
                model.fit(X_train, lengths=train_lengths)
                ll = model.score(X_val, lengths=val_lengths)
            except Exception as e:
                print(f"  n_components={n_comp} restart={r} failed: {e}")
                continue
            if ll > best_ll:
                best_ll = ll
                best_model = model
        results[n_comp] = (best_model, best_ll)
        print(f"n_components={n_comp}: val log-likelihood={best_ll:.1f}")

    best_n = max(results, key=lambda k: results[k][1])
    print(f"\nSelected n_components={best_n}")
    return best_n, results


# ---------------------------------------------------------------------------
# 5. Fit final model on all data with chosen n_components
# ---------------------------------------------------------------------------
def fit_final_model(X, lengths, n_components, n_iter=300,
                     covariance_type="diag", random_state=42, n_restarts=5):
    best_ll = -np.inf
    best_model = None
    for r in range(n_restarts):
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state + r,
            verbose=False,
        )
        model.fit(X, lengths=lengths)
        ll = model.score(X, lengths=lengths)
        if ll > best_ll:
            best_ll = ll
            best_model = model
    print(f"Final model log-likelihood: {best_ll:.1f}")
    return best_model


# ---------------------------------------------------------------------------
# 6. Inspect dwell times from the transition matrix
# ---------------------------------------------------------------------------
def inspect_dwell_times(model):
    diag = np.diag(model.transmat_)
    diag_clipped = np.clip(diag, 0, 0.999999)  # avoid divide-by-zero
    expected_dwell = 1 / (1 - diag_clipped)
    for k in range(model.n_components):
        print(f"state {k}: self-trans={diag[k]:.3f}, "
              f"expected dwell={expected_dwell[k]:.1f} frames")
    return expected_dwell


# ---------------------------------------------------------------------------
# 7. Merge redundant states via hierarchical clustering on state means
# ---------------------------------------------------------------------------
def merge_states(model, n_motifs, metric="cosine", method="average"):
    means = model.means_  # (n_components, D)
    dist = pdist(means, metric=metric)
    Z = linkage(dist, method=method)
    merged_labels = fcluster(Z, t=n_motifs, criterion="maxclust") - 1  # 0-indexed
    # merged_labels[k] gives the motif id for original HMM state k
    return merged_labels


# ---------------------------------------------------------------------------
# 8. Decode states per sequence + temporal smoothing + remap to motifs
# ---------------------------------------------------------------------------
def decode_and_smooth(model, X, lengths, merged_labels=None, smooth_window=5):
    raw_states = model.predict(X, lengths=lengths)  # (N,) HMM state ids

    if merged_labels is not None:
        states = merged_labels[raw_states]
    else:
        states = raw_states

    # smooth per-sequence (don't smooth across sequence boundaries)
    offsets = np.cumsum([0] + list(lengths))
    smoothed = np.empty_like(states)
    for i in range(len(lengths)):
        seg = states[offsets[i]:offsets[i + 1]]
        smoothed[offsets[i]:offsets[i + 1]] = median_filter(seg, size=smooth_window)

    return raw_states, smoothed


# ---------------------------------------------------------------------------
# Example end-to-end usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- replace with your actual data ---
    # data = np.load("your_data.npy")  # shape (num_sequence, L, D)
    # seq_lengths = None  # or a list of true per-sequence lengths if padded
    data = np.load("your_data.npy")
    seq_lengths = None

    STRIDE = 2
    N_PCA = 15

    # 1. subsample
    X_raw, lengths = subsample_sequences(data, stride=STRIDE, seq_lengths=seq_lengths)
    print(f"Total subsampled frames: {len(X_raw)}, num sequences: {len(lengths)}")

    # 2. preprocess
    X_pca, scaler, pca = preprocess(X_raw, n_pca=N_PCA)

    # 3. split for model selection
    X_train, train_lengths, X_val, val_lengths = split_by_sequence(X_pca, lengths)

    # 4. select n_components
    best_n, results = select_n_components(
        X_train, train_lengths, X_val, val_lengths,
        candidates=(10, 20, 30, 50),
    )

    # 5. fit final model on full data
    model = fit_final_model(X_pca, lengths, n_components=best_n)

    # 6. inspect dwell times
    inspect_dwell_times(model)

    # 7. merge to a target number of interpretable motifs
    N_MOTIFS = 20
    merged_labels = merge_states(model, n_motifs=N_MOTIFS)

    # 8. decode + smooth
    raw_states, smoothed_states = decode_and_smooth(
        model, X_pca, lengths, merged_labels=merged_labels, smooth_window=5
    )

    # split smoothed_states back per original sequence if needed
    offsets = np.cumsum([0] + list(lengths))
    per_seq_states = [smoothed_states[offsets[i]:offsets[i + 1]]
                       for i in range(len(lengths))]

    np.save("hmm_motif_labels.npy", np.array(per_seq_states, dtype=object))
    print("Done. Saved per-sequence motif labels to hmm_motif_labels.npy")