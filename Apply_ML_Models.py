from __future__ import annotations

import argparse
import heapq
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import torch
from sentence_transformers import SentenceTransformer
import time

start = time.time()

# Path to data here.
CSV_PATH = r"DPE_concat_last_90_days.csv"


# Column selection for ML models. Also known as feature construction.
TEXT_COLS = [
    "Timestamp",
    "DeviceName",
    "AccountName",
    "InitiatingProcessAccountName",
    "FileName",
    "ProcessCommandLine",
    "ProcessIntegrityLevel",
    "InitiatingProcessFileName",
    "InitiatingProcessCommandLine",
    "ProcessTokenElevation",
    "SHA256",
    "InitiatingProcessSHA256",
]

# Data preparation for models starts here.
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x)
    if len(s) > 2000:
        s = s[:2000] + "…"
    return s


def ensure_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    for col in TEXT_COLS:
        if col not in chunk.columns:
            chunk[col] = ""
    return chunk


def record_to_text(rec: Dict[str, Any]) -> str:
    return " | ".join([f"{c}={safe_str(rec.get(c, ''))}" for c in TEXT_COLS])

def reservoir_sample_texts(
    csv_path: Path,
    sample_n: int,
    chunksize: int,
    seed: int,
    encoding: str | None,
) -> List[str]:
    rng = np.random.default_rng(seed)
    reservoir: List[str] = []
    seen = 0

    usecols = TEXT_COLS

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        low_memory=False,
        encoding=encoding,
        usecols=usecols,
    ):
        chunk = ensure_columns(chunk)
        records = chunk.to_dict(orient="records")

        for rec in records:
            txt = record_to_text(rec)
            seen += 1

            if len(reservoir) < sample_n:
                reservoir.append(txt)
            else:
                j = int(rng.integers(0, seen))
                if j < sample_n:
                    reservoir[j] = txt

    if not reservoir:
        raise RuntimeError("No rows read from CSV—check file path/format.")
    return reservoir


def score_stream_topk(
    csv_path: Path,
    model: SentenceTransformer,
    iso: IsolationForest,
    topk: int,
    chunksize: int,
    batch_size: int,
    encoding: str | None,
) -> List[Tuple[float, Dict[str, Any]]]:
    # Keep TOP-K smallest scores (most anomalous).
    # Use a heap of size K, storing (neg_score, tie, row) where neg_score = -score.
    # Most anomalous => lowest score => highest neg_score.
    heap: List[Tuple[float, int, Dict[str, Any]]] = []
    tie = 0

    usecols = TEXT_COLS

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        low_memory=False,
        encoding=encoding,
        usecols=usecols,
    ):
        chunk = ensure_columns(chunk)
        records = chunk.to_dict(orient="records")

        texts = [record_to_text(rec) for rec in records]

        X = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = iso.decision_function(X)  # higher=normal, lower=anomalous

        for rec, score in zip(records, scores):
            score_f = float(score)
            rec_out = dict(rec)
            rec_out["anomaly_score"] = score_f

            neg_score = -score_f  # more anomalous => larger neg_score

            if len(heap) < topk:
                heapq.heappush(heap, (neg_score, tie, rec_out))
            else:
                # heap[0] is smallest neg_score => least anomalous among kept
                if neg_score > heap[0][0]:
                    heapq.heapreplace(heap, (neg_score, tie, rec_out))

            tie += 1

    results = [(-neg_score, row_dict) for (neg_score, _, row_dict) in heap]
    results.sort(key=lambda t: t[0])  # score ascending
    return results


# Model parameter selection begins here.
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out",
        default="Top_1k_anomalies_miniLM_150kTraining2.csv",
        help="Output CSV for TOP-K anomalies",
    )

    # Main model parameter selection.
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--sample", type=int, default=150000, help="Training sample size 50k–300k.")
    ap.add_argument("--topk", type=int, default=1000, help="How many anomalies to keep")
    ap.add_argument("--contamination", type=float, default=0.01, help="Expected anomaly rate.")
    ap.add_argument("--chunksize", type=int, default=20000, help="CSV rows per chunk")
    ap.add_argument("--batch", type=int, default=256, help="Embedding batch size. Reduce if not enough memory.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed") # This is default setting.
    ap.add_argument("--encoding", default=None, help="CSV encoding override, e.g. utf-8, latin1") # How CSV is formatted
    args = ap.parse_args()

    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    # Check if NVIDIA GPU is available. Otherwise it uses the CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Torch device: {device}")

    # Bert begins here.
    print(f"[+] Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model, device=device)

    # BERT training begins here.
    print(f"[+] Pass 1: streaming reservoir sample (n={args.sample}) from {csv_path} ...")
    train_texts = reservoir_sample_texts(
        csv_path=csv_path,
        sample_n=args.sample,
        chunksize=args.chunksize,
        seed=args.seed,
        encoding=args.encoding,
    )


    # Bert embedding.
    print("[+] Embedding training sample...")
    X_train = model.encode(
        train_texts,
        batch_size=args.batch,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Isolation Forest training begins here.
    print("[+] Training IsolationForest...")
    iso = IsolationForest(
        n_estimators=400,
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=-1,
    )
    iso.fit(X_train)

    # Score the data and keep top anamolies.
    print(f"[+] Pass 2: scoring stream, keeping TOP-K={args.topk}...")
    top = score_stream_topk(
        csv_path=csv_path,
        model=model,
        iso=iso,
        topk=args.topk,
        chunksize=args.chunksize,
        batch_size=args.batch,
        encoding=args.encoding,
    )

    out_rows = [row for (_score, row) in top]
    out_df = pd.DataFrame(out_rows).sort_values("anomaly_score", ascending=True)

    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"[+] Wrote anomalies: {out_path.resolve()}")

    end = time.time()

    print(f"Execution time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()