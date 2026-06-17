"""
DCR sweep over M1 batch_size x max_epochs.
4 configs in parallel across 4 GPUs. Each process writes its own result file.
"""
import os, sys, json, shutil, warnings, tempfile, multiprocessing as mp
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

DATA_DIR  = Path("/shared/diabetes-synth/data/splits")
RESULTS_F = Path("/shared/diabetes-synth/results/m1_sweep_dcr.json")
TMP_DIR   = Path("/shared/diabetes-synth/results/sweep_tmp")
DCR_SAMPLE = 5_000
N_FOLDS    = 5
TARGET     = "diabetes"

BATCH_SIZES = [64, 256, 512, 1024]
MAX_EPOCHS  = [5, 10, 15, 50]
CONFIGS = [(bs, ep) for bs in BATCH_SIZES for ep in MAX_EPOCHS
           if not (bs == 64 and ep == 50)]


def _load_fold(fold):
    import pandas as pd
    train = pd.read_csv(DATA_DIR / str(fold) / "train.csv")
    test  = pd.read_csv(DATA_DIR / str(fold) / "test.csv")
    return train, test


def _encode(df, encoders=None, fit=False):
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    cats = ["gender", "smoking_history"]
    if fit:
        encoders = {}
        for col in cats:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        return df, encoders
    for col in cats:
        le = encoders[col]
        df[col] = df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
    return df, encoders


def _worker(batch_size, max_epochs, gpu_id, out_file):
    """Runs on one GPU, trains all 5 folds, writes result JSON to out_file."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        warnings.filterwarnings("ignore")

        import numpy as np, shutil, tempfile
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        from mostlyai.engine import TabularARGN

        fold_results = []
        for fold in range(N_FOLDS):
            train_df, test_df = _load_fold(fold)
            train_df, encoders = _encode(train_df, fit=True)
            test_df,  _        = _encode(test_df, encoders=encoders, fit=False)

            m1_data = train_df[train_df[TARGET] == 1].drop(columns=[TARGET])
            feats   = [c for c in train_df.columns if c != TARGET]
            scaler  = StandardScaler()
            X_tr_s  = scaler.fit_transform(train_df[feats].values.astype(float))
            X_te_s  = scaler.transform(test_df[feats].values.astype(float))

            ws = tempfile.mkdtemp(prefix=f"sw_b{batch_size}_e{max_epochs}_f{fold}_")
            try:
                argn = TabularARGN(
                    model="MOSTLY_AI/Medium",
                    max_epochs=float(max_epochs),
                    batch_size=None if batch_size == 64 else batch_size,
                    workspace_dir=ws,
                    device="cuda:0",
                    verbose=0,
                )
                argn.fit(m1_data)

                prog = Path(ws) / "ModelStore/model-data/progress-messages.csv"
                actual_epochs = None
                if prog.exists():
                    rows = prog.read_text().strip().split("\n")
                    if len(rows) > 1:
                        actual_epochs = float(rows[-1].split(",")[0])

                syn = argn.sample(n_samples=DCR_SAMPLE)
                syn, _ = _encode(syn, encoders=encoders, fit=False)
                X_syn = scaler.transform(syn[feats].values.astype(float))

                d_tr, _ = NearestNeighbors(n_neighbors=1, n_jobs=4).fit(X_tr_s).kneighbors(X_syn)
                d_te, _ = NearestNeighbors(n_neighbors=1, n_jobs=4).fit(X_te_s).kneighbors(X_syn)
                ratio   = float(d_tr.mean() / d_te.mean())

                fold_results.append({
                    "fold": fold, "actual_epochs": actual_epochs,
                    "dcr_train": round(float(d_tr.mean()), 4),
                    "dcr_test":  round(float(d_te.mean()), 4),
                    "ratio":     round(ratio, 3),
                })
                print(f"  [b={batch_size} e={max_epochs} f={fold}] "
                      f"epochs={actual_epochs}  ratio={ratio:.3f}", flush=True)
            finally:
                shutil.rmtree(ws, ignore_errors=True)

        avg = round(float(np.mean([r["ratio"] for r in fold_results])), 3)
        result = {"batch_size": batch_size, "max_epochs": max_epochs,
                  "avg_ratio": avg, "splits": fold_results}
        Path(out_file).write_text(json.dumps(result, indent=2))
        print(f"  DONE b={batch_size} e={max_epochs} avg_ratio={avg}", flush=True)

    except Exception as exc:
        import traceback
        Path(out_file).write_text(json.dumps(
            {"batch_size": batch_size, "max_epochs": max_epochs,
             "error": str(exc), "traceback": traceback.format_exc()}
        ))
        print(f"  ERROR b={batch_size} e={max_epochs}: {exc}", flush=True)


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_F.parent.mkdir(parents=True, exist_ok=True)

    all_results = [{"batch_size": 64, "max_epochs": 50,
                    "avg_ratio": 0.713, "note": "original Medium — reference"}]

    ctx = mp.get_context("spawn")

    for i in range(0, len(CONFIGS), 4):
        batch = CONFIGS[i: i + 4]
        print(f"\n{'='*60}\nRound {i//4+1}: {batch}\n{'='*60}", flush=True)

        procs, out_files = [], []
        for gpu_idx, (bs, ep) in enumerate(batch):
            out_f = str(TMP_DIR / f"b{bs}_e{ep}.json")
            out_files.append((out_f, bs, ep))
            p = ctx.Process(target=_worker, args=(bs, ep, gpu_idx, out_f))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        for out_f, bs, ep in out_files:
            if Path(out_f).exists():
                r = json.loads(Path(out_f).read_text())
                all_results.append(r)
                if "error" in r:
                    print(f"  FAILED b={bs} e={ep}: {r['error']}")
            else:
                print(f"  NO OUTPUT for b={bs} e={ep}")

        RESULTS_F.write_text(json.dumps(all_results, indent=2))

    print(f"\n{'='*60}\nSWEEP COMPLETE — ranked by avg_ratio:\n{'='*60}")
    for r in sorted(all_results, key=lambda x: -x.get("avg_ratio", 0)):
        if "error" not in r:
            print(f"  batch={r['batch_size']:4d}  epochs={r['max_epochs']:2d}  "
                  f"ratio={r['avg_ratio']:.3f}  {r.get('note','')}")


if __name__ == "__main__":
    main()
