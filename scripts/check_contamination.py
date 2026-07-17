#!/usr/bin/env python3
"""Split-aware contamination / overlap check between ViRL39K and evaluation splits."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator

import yaml

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

INSTRUCTION_SUFFIX = re.compile(
    r"\n?\s*You first think through the reasoning process.*?\\boxed\{\}\.\s*$",
    re.DOTALL | re.IGNORECASE,
)
CHOICES_BRACKET = re.compile(r"\[([^\]]+)\]")
CHOICES_BLOCK = re.compile(r"Choices:\s*(.+?)(?:\n\n|\Z)", re.DOTALL | re.IGNORECASE)

EVAL_SPLITS = frozenset(
    {"test", "testmini", "validation", "val", "dev", "dev-as-test", "eval", "evaluation"}
)
TRAIN_SPLITS = frozenset({"train", "training"})


class RunLogger:
    """Timestamped log to stdout and run.log."""

    def __init__(self, out_dir: Path, log_name: str = "run.log") -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = out_dir / log_name
        self.status_path = out_dir / "latest_status.txt"
        self.info(f"Log file: {self.log_path}")

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def info(self, msg: str) -> None:
        line = f"[{self._ts()}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.status_path.write_text(line + "\n", encoding="utf-8")

    def warn(self, msg: str) -> None:
        self.info(f"WARN: {msg}")


class CheckpointManager:
    """Incremental progress + image hash cache under {out}/checkpoints/."""

    def __init__(self, out_dir: Path, logger: RunLogger) -> None:
        self.dir = out_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.dir / "progress.json"
        self.image_cache_path = self.dir / "image_hash_cache.jsonl"
        self.logger = logger
        self.progress = self._load_progress()
        self.image_cache: dict[str, tuple[str, str]] = self._load_image_cache()

    def _load_progress(self) -> dict[str, Any]:
        if self.progress_path.is_file():
            with open(self.progress_path, encoding="utf-8") as f:
                return json.load(f)
        return {"completed_benchmarks": [], "started_at": datetime.now(timezone.utc).isoformat()}

    def _save_progress(self, phase: str) -> None:
        self.progress["phase"] = phase
        self.progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.progress_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)
        tmp.replace(self.progress_path)

    def _load_image_cache(self) -> dict[str, tuple[str, str]]:
        cache: dict[str, tuple[str, str]] = {}
        if not self.image_cache_path.is_file():
            return cache
        with open(self.image_cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cache[row["path"]] = (row.get("sha256", ""), row.get("phash", ""))
        self.logger.info(f"Loaded {len(cache)} cached image hashes from {self.image_cache_path}")
        return cache

    def append_image_cache(self, path: str, sha256: str, phash: str) -> None:
        self.image_cache[path] = (sha256, phash)
        with open(self.image_cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"path": path, "sha256": sha256, "phash": phash}, ensure_ascii=False) + "\n")

    def is_benchmark_done(self, name: str) -> bool:
        return name in self.progress.get("completed_benchmarks", [])

    def mark_benchmark_done(self, name: str) -> None:
        done = self.progress.setdefault("completed_benchmarks", [])
        if name not in done:
            done.append(name)
        self._save_progress(f"done:{name}")

    def save_benchmark_artifact(self, name: str, rows: list[OverlapRow], stat: dict[str, Any]) -> None:
        bench_dir = self.dir / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        write_csv(bench_dir / f"{name}_overlap.csv", rows)
        tmp = bench_dir / f"{name}_stats.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stat, f, ensure_ascii=False, indent=2)
        tmp.replace(bench_dir / f"{name}_stats.json")

    def save_partial_summary(
        self,
        out_dir: Path,
        stats: list[dict[str, Any]],
        all_overlap: list[OverlapRow],
        warnings_out: list[str],
        train_count: int,
    ) -> None:
        confirmed = [r for r in all_overlap if r.classification == "confirmed_eval_overlap"]
        total_confirmed_ids = len({r.eval_id for r in confirmed})
        total_eval = sum(s["n_eval_samples"] for s in stats)
        partial = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "phase": self.progress.get("phase", ""),
            "completed_benchmarks": self.progress.get("completed_benchmarks", []),
            "train_samples": train_count,
            "per_benchmark": stats,
            "totals": {
                "eval_samples": total_eval,
                "confirmed_eval_overlap_unique_eval_ids": total_confirmed_ids,
                "overlap_candidates": len(
                    [r for r in all_overlap if r.classification != "benign_train_split_overlap"]
                ),
            },
            "warnings_tail": warnings_out[-20:],
        }
        write_json_atomic(out_dir / "summary.partial.json", partial)

    def load_completed_rows(self) -> list[OverlapRow]:
        rows: list[OverlapRow] = []
        bench_dir = self.dir / "benchmarks"
        if not bench_dir.is_dir():
            return rows
        for name in self.progress.get("completed_benchmarks", []):
            path = bench_dir / f"{name}_overlap.csv"
            if not path.is_file():
                continue
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(
                        OverlapRow(
                            candidate_type=row["candidate_type"],
                            classification=row["classification"],
                            eval_dataset=row["eval_dataset"],
                            eval_split=row["eval_split"],
                            eval_id=row["eval_id"],
                            train_id=row["train_id"],
                            similarity=float(row["similarity"]),
                            eval_question=row["eval_question"],
                            train_question=row["train_question"],
                            eval_answer=row["eval_answer"],
                            train_answer=row["train_answer"],
                            image_hash_match=row["image_hash_match"] in {"True", "true", "1"},
                            phash_distance=(
                                int(row["phash_distance"])
                                if row.get("phash_distance") not in (None, "", "None")
                                else None
                            ),
                            match_reason=row["match_reason"],
                            source_dataset=row.get("source_dataset", ""),
                            train_source_split=row.get("train_source_split", ""),
                        )
                    )
        return rows

    def load_completed_stats(self) -> list[dict[str, Any]]:
        stats: list[dict[str, Any]] = []
        bench_dir = self.dir / "benchmarks"
        for name in self.progress.get("completed_benchmarks", []):
            path = bench_dir / f"{name}_stats.json"
            if path.is_file():
                with open(path, encoding="utf-8") as f:
                    stats.append(json.load(f))
        return stats


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def flush_aggregate_outputs(
    out_dir: Path,
    all_overlap: list[OverlapRow],
) -> None:
    """Rewrite cumulative CSVs after each benchmark (incremental checkpoint)."""
    confirmed = [r for r in all_overlap if r.classification == "confirmed_eval_overlap"]
    suspicious = [r for r in all_overlap if r.classification == "suspicious_near_duplicate"]
    candidates = [r for r in all_overlap if r.classification != "benign_train_split_overlap"]
    image_rows = [r for r in all_overlap if r.candidate_type in {"image_sha256", "phash_overlap"}]
    write_csv(out_dir / "overlap_candidates.csv", candidates)
    write_csv(out_dir / "confirmed_eval_overlap.csv", confirmed)
    write_csv(out_dir / "near_duplicate_candidates.csv", suspicious)
    write_csv(out_dir / "image_overlap_candidates.csv", image_rows)


@dataclass
class ManifestRecord:
    record_type: str  # train | eval
    dataset_name: str
    split_name: str
    record_id: str
    original_id: str
    source_dataset: str
    source_split: str
    source_original_id: str
    question: str
    choices: str
    answer: str
    image_path_or_url: str
    normalized_question: str
    normalized_question_choices: str
    normalized_question_choices_answer: str
    image_sha256: str = ""
    image_phash: str = ""
    field_map_note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OverlapRow:
    candidate_type: str
    classification: str
    eval_dataset: str
    eval_split: str
    eval_id: str
    train_id: str
    similarity: float
    eval_question: str
    train_question: str
    eval_answer: str
    train_answer: str
    image_hash_match: bool
    phash_distance: int | None
    match_reason: str
    source_dataset: str = ""
    train_source_split: str = ""


# ---------------------------------------------------------------------------
# Normalization & hashing
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = re.sub(r"<image>", "", text, flags=re.IGNORECASE)
    text = INSTRUCTION_SUFFIX.sub("", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\$\{\}\\\(\)\[\]\.\,\;\:\+\-\=\^\°\%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_choices(question: str) -> str:
    m = CHOICES_BLOCK.search(question)
    if m:
        return m.group(1).strip()
    m = CHOICES_BRACKET.search(question)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in question.splitlines() if re.match(r"^[A-Da-d][\.\)]\s", ln.strip())]
    return " | ".join(lines)


def strip_boxed(answer: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", answer)
    return m.group(1).strip() if m else answer.strip()


def average_phash(image_path: Path) -> str:
    if not HAS_PIL or not image_path.is_file():
        return ""
    with Image.open(image_path) as im:
        im = im.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(im.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return bits


def phash_distance(a: str, b: str) -> int | None:
    if not a or not b or len(a) != len(b):
        return None
    return sum(x != y for x, y in zip(a, b))


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_image_path(path_str: str) -> tuple[str, str, str]:
    """ProcessPool worker: path -> (path, sha256, phash)."""
    p = Path(path_str)
    if not p.is_file():
        return path_str, "", ""
    return path_str, sha256_file(p), average_phash(p)


@dataclass
class TrainNearDupIndex:
    train_records: list[ManifestRecord]
    vectorizer: Any
    X_train: Any


# ---------------------------------------------------------------------------
# Flexible field resolver
# ---------------------------------------------------------------------------

ID_KEYS = ("id", "question_id", "sample_id", "uid", "qid", "original_id", "source_id")
DATASET_KEYS = ("dataset", "source_dataset", "dataset_name", "source")
SPLIT_KEYS = ("split", "source_split", "subset")
QUESTION_KEYS = ("question", "query", "problem", "prompt", "instruction")
ANSWER_KEYS = ("answer", "label", "gt_answer", "ground_truth", "response")
CHOICES_KEYS = ("choices", "options")
IMAGE_KEYS = ("image", "image_path", "image_url", "img", "images")


def first_key(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in row and row[k] not in (None, "", []):
            return k
    return None


def resolve_image_path(raw: Any, root: Path) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    if not isinstance(raw, str) or not raw:
        return ""
    p = raw[2:] if raw.startswith("./") else raw
    return str((root / p).resolve()) if not Path(p).is_absolute() else p


def parse_sharegpt(row: dict[str, Any]) -> tuple[str, str, str]:
    question, answer, choices = "", "", ""
    for msg in row.get("messages") or []:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role == "user":
            question = content
            choices = extract_choices(content)
        elif role == "assistant":
            answer = content
    return question, choices, answer


def build_record(
    row: dict[str, Any],
    *,
    record_type: str,
    dataset_name: str,
    split_name: str,
    record_id: str,
    image_root: Path,
    field_notes: dict[str, str],
    source_dataset: str = "",
    source_split: str = "",
    source_original_id: str = "",
    original_id: str = "",
) -> ManifestRecord:
    if "messages" in row:
        question, choices, answer = parse_sharegpt(row)
        field_notes.setdefault("format", "sharegpt")
        img_key = first_key(row, IMAGE_KEYS) or "images"
        image_raw = row.get(img_key)
    else:
        qk = first_key(row, QUESTION_KEYS)
        ak = first_key(row, ANSWER_KEYS)
        ck = first_key(row, CHOICES_KEYS)
        question = str(row.get(qk, "")) if qk else ""
        choices = str(row.get(ck, "")) if ck else extract_choices(question)
        answer = str(row.get(ak, "")) if ak else ""
        for k in (qk, ak, ck, first_key(row, IMAGE_KEYS)):
            if k:
                field_notes[k] = k
        img_key = first_key(row, IMAGE_KEYS)
        image_raw = row.get(img_key) if img_key else ""

    answer_clean = strip_boxed(answer)
    nq = normalize_text(question)
    nqc = normalize_text(f"{question} {choices}".strip())
    nqca = normalize_text(f"{question} {choices} {answer_clean}".strip())
    image_path = resolve_image_path(image_raw, image_root)

    return ManifestRecord(
        record_type=record_type,
        dataset_name=dataset_name,
        split_name=split_name,
        record_id=record_id,
        original_id=original_id or record_id,
        source_dataset=source_dataset,
        source_split=source_split,
        source_original_id=source_original_id,
        question=question,
        choices=choices,
        answer=answer_clean,
        image_path_or_url=image_path,
        normalized_question=nq,
        normalized_question_choices=nqc,
        normalized_question_choices_answer=nqca,
        field_map_note=json.dumps(field_notes, ensure_ascii=False),
    )


def enrich_image_hashes(
    records: list[ManifestRecord],
    warnings_out: list[str],
    cache: dict[str, tuple[str, str]],
    checkpoint: CheckpointManager | None = None,
    logger: RunLogger | None = None,
    log_interval: int = 500,
    label: str = "images",
    workers: int = 1,
) -> None:
    pending = [
        rec
        for rec in records
        if rec.image_path_or_url and str(Path(rec.image_path_or_url)) not in cache
    ]
    total_new = len(pending)
    if logger:
        worker_note = f", workers={workers}" if workers > 1 else ""
        logger.info(
            f"{label}: {total_new} new hashes to compute ({len(cache)} already cached{worker_note})"
        )
    t0 = time.time()

    def store_hash(key: str, sha: str, phash: str, i: int) -> None:
        if not sha and not phash and key and not Path(key).is_file():
            warnings_out.append(f"Missing image: {key}")
        cache[key] = (sha, phash)
        if checkpoint:
            checkpoint.append_image_cache(key, sha, phash)
        if logger and (i == 1 or i % log_interval == 0 or i == total_new):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            logger.info(
                f"{label}: {i}/{total_new} new hashed ({rate:.1f} img/s, cache size {len(cache)})"
            )

    if workers > 1 and total_new > 0:
        paths = [str(Path(rec.image_path_or_url)) for rec in pending]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_hash_image_path, p) for p in paths]
            for i, fut in enumerate(as_completed(futures), start=1):
                key, sha, phash = fut.result()
                store_hash(key, sha, phash, i)
    else:
        for i, rec in enumerate(pending, start=1):
            p = Path(rec.image_path_or_url)
            key = str(p)
            if not p.is_file():
                store_hash(key, "", "", i)
            else:
                store_hash(key, sha256_file(p), average_phash(p), i)

    for rec in records:
        if not rec.image_path_or_url:
            continue
        key = str(Path(rec.image_path_or_url))
        rec.image_sha256, rec.image_phash = cache.get(key, ("", ""))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def iter_json_array(path: Path, max_samples: int | None) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    for i, row in enumerate(data):
        if max_samples is not None and i >= max_samples:
            break
        yield row


def iter_jsonl(path: Path, max_samples: int | None) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_parquet(path: Path, max_samples: int | None) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "Reading .parquet requires pyarrow (e.g. conda activate fast_grpo)."
        ) from e

    pf = pq.ParquetFile(path)
    seen = 0
    for batch in pf.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            if max_samples is not None and seen >= max_samples:
                return
            yield row
            seen += 1


def load_rows(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf == ".json":
        return list(iter_json_array(path, max_samples))
    if suf == ".jsonl":
        return list(iter_jsonl(path, max_samples))
    if suf == ".parquet":
        return list(iter_parquet(path, max_samples))
    if suf in {".csv", ".tsv"}:
        import csv as csvmod

        delim = "\t" if suf == ".tsv" else ","
        rows = []
        with open(path, encoding="utf-8") as f:
            reader = csvmod.DictReader(f, delimiter=delim)
            for i, row in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break
                rows.append(row)
        return rows
    raise ValueError(f"Unsupported file type: {path}")


def load_id_column(path: Path, id_col: str = "id") -> list[str]:
    rows = load_rows(path, None)
    out = []
    for r in rows:
        if id_col in r:
            out.append(str(r[id_col]))
        elif "messages" in r and "id" in r:
            out.append(str(r["id"]))
    return out


def parse_train_qid(qid: str, source: str, aliases: dict[str, str | None]) -> tuple[str, str, str]:
    canonical = aliases.get(source, source)
    if canonical is None:
        canonical = source
    original = qid
    for prefix in (f"{source}-", f"{canonical}-", "MMK12-", "Processed-", "geoqa_plus-"):
        if qid.startswith(prefix):
            original = qid[len(prefix) :]
            break
    return canonical, "", original


def load_train_manifest(
    train_root: Path,
    train_cfg: dict[str, Any],
    aliases: dict[str, str | None],
    max_samples: int | None,
    warnings_out: list[str],
) -> list[ManifestRecord]:
    rel = train_cfg.get("path", "virl39k_train.json")
    path = train_root / rel
    if not path.is_file():
        fb = train_cfg.get("parquet_fallback")
        if fb and (train_root / fb).is_file():
            path = train_root / fb
            warnings_out.append(f"Using parquet fallback for train: {path}")
        else:
            raise FileNotFoundError(f"Train manifest not found: {path}")

    image_root = train_root / train_cfg.get("image_root", ".")
    records: list[ManifestRecord] = []
    rows = load_rows(path, max_samples)
    field_notes = {"train_file": str(path)}

    for i, row in enumerate(rows):
        source = str(row.get("source", ""))
        qid = str(row.get("qid", row.get("id", f"train_{i}")))
        sd, ss, soid = parse_train_qid(qid, source, aliases)
        rec = build_record(
            row,
            record_type="train",
            dataset_name="ViRL39K",
            split_name="train",
            record_id=f"train_{i}",
            image_root=image_root,
            field_notes=dict(field_notes),
            source_dataset=sd,
            source_split=ss,
            source_original_id=soid,
            original_id=qid,
        )
        records.append(rec)
    return records


def load_eval_manifest(
    eval_root: Path,
    name: str,
    cfg: dict[str, Any],
    max_samples: int | None,
    warnings_out: list[str],
) -> list[ManifestRecord]:
    path = Path(cfg["path"])
    if not path.is_absolute():
        path = eval_root / path
    if not path.is_file():
        warnings_out.append(f"[SKIP] Eval file missing for {name}: {path}")
        return []

    image_root = eval_root / cfg.get("image_root", ".")
    rows = load_rows(path, max_samples)
    test_ids: list[str] = []
    tp = cfg.get("test_id_parquet")
    if tp and Path(tp).is_file():
        test_ids = load_id_column(Path(tp), "id")
        if len(test_ids) != len(rows):
            warnings_out.append(
                f"{name}: test_id_parquet length {len(test_ids)} != eval rows {len(rows)}; index-align while possible"
            )

    records = []
    field_notes = {"eval_file": str(path), "format": "sharegpt" if rows and "messages" in rows[0] else "generic"}
    split_name = cfg.get("split", "test")
    dataset_name = cfg.get("dataset_name", name)
    source_dataset = cfg.get("source_dataset", dataset_name)

    for i, row in enumerate(rows):
        oid = test_ids[i] if i < len(test_ids) else str(row.get("id", f"test_{i}"))
        rec = build_record(
            row,
            record_type="eval",
            dataset_name=dataset_name,
            split_name=split_name,
            record_id=f"{dataset_name}_{i}",
            image_root=image_root,
            field_notes=dict(field_notes),
            source_dataset=source_dataset,
            source_split=split_name,
            source_original_id=oid,
            original_id=oid,
        )
        records.append(rec)
    return records


def load_reference_id_sets(cfg: dict[str, Any], warnings_out: list[str]) -> tuple[set[str], set[str]]:
    test_ids: set[str] = set()
    train_ids: set[str] = set()
    tp = cfg.get("test_id_parquet")
    if tp and Path(tp).is_file():
        test_ids.update(load_id_column(Path(tp), "id"))
    tg = cfg.get("train_id_parquet_glob")
    if tg:
        for fp in sorted(glob.glob(tg)):
            try:
                train_ids.update(load_id_column(Path(fp), "id"))
            except Exception as e:
                warnings_out.append(f"Failed loading train ids from {fp}: {e}")
    return test_ids, train_ids


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------


def is_template_question(nq: str) -> bool:
    if len(nq) < 15:
        return True
    templates = (
        "what is the answer",
        "how many",
        "choose the correct",
        "select the correct option",
    )
    return any(t in nq for t in templates) and len(nq) < 40


def classify_overlap(
    row: OverlapRow,
    *,
    eval_split: str,
    train_source_split: str,
    metadata_eval_hit: bool,
    metadata_train_hit: bool,
    text_threshold: float,
    phash_strong: int,
) -> str:
    if metadata_train_hit and not metadata_eval_hit:
        return "benign_train_split_overlap"
    if metadata_eval_hit:
        return "confirmed_eval_overlap"
    if row.image_hash_match and row.similarity >= 1.0:
        return "confirmed_eval_overlap"
    if row.image_hash_match and row.similarity >= text_threshold and row.eval_answer == row.train_answer:
        return "confirmed_eval_overlap"
    if row.similarity >= 0.95 and row.image_hash_match:
        if row.phash_distance is not None and row.phash_distance <= phash_strong:
            return "confirmed_eval_overlap"
        return "confirmed_eval_overlap"
    if row.image_hash_match and row.similarity < 0.5:
        return "uncertain_manual_review"
    if is_template_question(row.eval_question) or is_template_question(row.train_question):
        return "uncertain_manual_review"
    if row.similarity >= text_threshold:
        return "suspicious_near_duplicate"
    if row.phash_distance is not None and row.phash_distance <= phash_strong and row.similarity >= 0.7:
        return "suspicious_near_duplicate"
    if row.image_hash_match:
        return "uncertain_manual_review"
    return "uncertain_manual_review"


def metadata_lookup(
    train: ManifestRecord,
    eval_rec: ManifestRecord,
    test_ids: set[str],
    train_ids: set[str],
) -> tuple[bool, bool]:
    if not train.source_original_id or not eval_rec.source_original_id:
        return False, False
    sd_train = train.source_dataset.lower().replace("_", "").replace("-", "")
    sd_eval = eval_rec.source_dataset.lower().replace("_", "").replace("-", "")
    if sd_train not in sd_eval and sd_eval not in sd_train and sd_train != sd_eval:
        # loose family match
        if not (sd_train in {"mmk12", "k12"} and sd_eval in {"mmk12", "k12"}):
            if sd_train != sd_eval:
                return False, False
    oid = train.source_original_id
    eval_hit = oid == eval_rec.source_original_id or oid in test_ids
    train_hit = oid in train_ids
    return eval_hit, train_hit


def run_checks(
    train_records: list[ManifestRecord],
    eval_records: list[ManifestRecord],
    eval_name: str,
    eval_cfg: dict[str, Any],
    *,
    text_threshold: float,
    phash_strong: int,
    phash_weak: int,
    topk: int,
    warnings_out: list[str],
    skip_near_duplicate: bool = False,
    logger: RunLogger | None = None,
    near_dup_index: TrainNearDupIndex | None = None,
) -> list[OverlapRow]:
    test_ids, train_ids = load_reference_id_sets(eval_cfg, warnings_out)
    eval_split = eval_cfg.get("split", "test")

    # Index train by normalized fields & image hash
    train_by_nqc: dict[str, list[ManifestRecord]] = defaultdict(list)
    train_by_img: dict[str, list[ManifestRecord]] = defaultdict(list)
    train_by_meta: dict[tuple[str, str], ManifestRecord] = {}
    for tr in train_records:
        if tr.normalized_question_choices:
            train_by_nqc[tr.normalized_question_choices].append(tr)
        if tr.image_sha256:
            train_by_img[tr.image_sha256].append(tr)
        if tr.source_dataset and tr.source_original_id:
            train_by_meta[(tr.source_dataset, tr.source_original_id)] = tr

    rows: list[OverlapRow] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    def add_row(candidate_type: str, er: ManifestRecord, tr: ManifestRecord, sim: float, reason: str) -> None:
        key = (er.record_id, tr.record_id, candidate_type)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        img_match = bool(er.image_sha256 and er.image_sha256 == tr.image_sha256)
        pdist = phash_distance(er.image_phash, tr.image_phash) if er.image_phash and tr.image_phash else None
        meta_eval, meta_train = metadata_lookup(tr, er, test_ids, train_ids)
        base = OverlapRow(
            candidate_type=candidate_type,
            classification="pending",
            eval_dataset=eval_name,
            eval_split=eval_split,
            eval_id=er.original_id,
            train_id=tr.original_id,
            similarity=sim,
            eval_question=er.question[:500],
            train_question=tr.question[:500],
            eval_answer=er.answer,
            train_answer=tr.answer,
            image_hash_match=img_match,
            phash_distance=pdist,
            match_reason=reason,
            source_dataset=tr.source_dataset,
            train_source_split=tr.source_split,
        )
        base.classification = classify_overlap(
            base,
            eval_split=eval_split,
            train_source_split=tr.source_split,
            metadata_eval_hit=meta_eval and er.original_id == tr.source_original_id
            or (tr.source_dataset == er.source_dataset and tr.source_original_id == er.original_id),
            metadata_train_hit=meta_train,
            text_threshold=text_threshold,
            phash_strong=phash_strong,
        )
        rows.append(base)

    # 1) Metadata exact ID via index (avoid O(train*eval))
    meta_index: dict[tuple[str, str], ManifestRecord] = {}
    for tr in train_records:
        if tr.source_original_id:
            keys = {(tr.source_dataset, tr.source_original_id)}
            if tr.source_dataset in {"MMK12", "K12"}:
                keys.add(("MMK12", tr.source_original_id))
            for key in keys:
                meta_index.setdefault(key, tr)

    for er in eval_records:
        if not er.source_original_id:
            continue
        keys = [(er.source_dataset, er.source_original_id)]
        if er.source_dataset == "MMK12":
            keys.append(("MMK12", er.source_original_id))
        for key in keys:
            tr = meta_index.get(key)
            if tr:
                add_row("metadata_exact_id", er, tr, 1.0, "source_original_id == eval original_id")
                break

    # 2) Exact normalized text
    for er in eval_records:
        if not er.normalized_question_choices:
            continue
        for tr in train_by_nqc.get(er.normalized_question_choices, []):
            add_row("exact_text_qc", er, tr, 1.0, "normalized_question_choices exact match")

    # 3) Image SHA256
    for er in eval_records:
        if not er.image_sha256:
            continue
        for tr in train_by_img.get(er.image_sha256, []):
            sim = SequenceMatcher(None, er.normalized_question_choices, tr.normalized_question_choices).ratio()
            add_row("image_sha256", er, tr, sim, "image_sha256 exact match")

    # 4) Near-duplicate retrieval (optional; slow on full train without sklearn)
    if not skip_near_duplicate and eval_records and train_records:
        if logger:
            logger.info(
                f"{eval_name}: near-duplicate retrieval on {len(eval_records)} eval x {len(train_records)} train"
            )
        t0 = time.time()
        eval_texts = [e.normalized_question_choices for e in eval_records]
        if near_dup_index is not None and any(eval_texts):
            if logger:
                logger.info(f"{eval_name}: cosine similarity (shared TF-IDF index)...")
            X_eval = near_dup_index.vectorizer.transform(eval_texts)
            sims = cosine_similarity(X_eval, near_dup_index.X_train)
            train_records_nd = near_dup_index.train_records
            for ei, er in enumerate(eval_records):
                scores = sims[ei]
                top_idx = scores.argsort()[::-1][:topk]
                for ti in top_idx:
                    sim = float(scores[ti])
                    if sim >= text_threshold:
                        add_row(
                            "near_duplicate_text",
                            er,
                            train_records_nd[ti],
                            sim,
                            "TF-IDF char cosine (shared index)",
                        )
            if logger:
                logger.info(f"{eval_name}: near-duplicate done in {time.time() - t0:.1f}s")
        elif HAS_SKLEARN and any(train_records) and any(eval_texts):
            train_texts = [t.normalized_question_choices for t in train_records]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            try:
                if logger:
                    logger.info(f"{eval_name}: TF-IDF fit_transform on {len(train_texts)} train texts...")
                X_train = vec.fit_transform(train_texts)
                X_eval = vec.transform(eval_texts)
                if logger:
                    logger.info(f"{eval_name}: cosine similarity {len(eval_texts)} x {len(train_texts)}...")
                sims = cosine_similarity(X_eval, X_train)
                for ei, er in enumerate(eval_records):
                    scores = sims[ei]
                    top_idx = scores.argsort()[::-1][:topk]
                    for ti in top_idx:
                        sim = float(scores[ti])
                        if sim >= text_threshold:
                            add_row("near_duplicate_text", er, train_records[ti], sim, "TF-IDF char cosine")
                if logger:
                    logger.info(f"{eval_name}: near-duplicate done in {time.time() - t0:.1f}s")
            except Exception as e:
                warnings_out.append(f"{eval_name}: TF-IDF failed ({e}); fallback difflib")
                if logger:
                    logger.warn(f"{eval_name}: TF-IDF failed, using difflib fallback")
        else:
            if logger:
                logger.info(f"{eval_name}: difflib fallback (sklearn unavailable or empty texts)")
            train_texts_list = train_records
            for ei, er in enumerate(eval_records):
                if logger and ei > 0 and ei % 200 == 0:
                    logger.info(f"{eval_name}: difflib progress {ei}/{len(eval_records)}")
                best_sim, best_tr = 0.0, None
                for tr in train_texts_list:
                    sim = SequenceMatcher(None, er.normalized_question_choices, tr.normalized_question_choices).ratio()
                    if sim > best_sim:
                        best_sim, best_tr = sim, tr
                if best_tr and best_sim >= text_threshold:
                    add_row("near_duplicate_text", er, best_tr, best_sim, "SequenceMatcher fallback")
            if logger:
                logger.info(f"{eval_name}: difflib fallback done in {time.time() - t0:.1f}s")

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[OverlapRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def aggregate_stats(
    eval_name: str,
    eval_cfg: dict[str, Any],
    eval_records: list[ManifestRecord],
    all_rows: list[OverlapRow],
) -> dict[str, Any]:
    n_eval = len(eval_records)
    by_cls = defaultdict(set)
    for r in all_rows:
        by_cls[r.classification].add(r.eval_id)
    confirmed = [r for r in all_rows if r.classification == "confirmed_eval_overlap"]
    return {
        "eval_benchmark": eval_name,
        "eval_split": eval_cfg.get("split", "test"),
        "n_eval_samples": n_eval,
        "exact_id_overlap": len({r.eval_id for r in all_rows if r.candidate_type == "metadata_exact_id"}),
        "exact_text_image_overlap": len(
            {
                r.eval_id
                for r in all_rows
                if r.candidate_type in {"exact_text_qc", "image_sha256"}
                and r.classification == "confirmed_eval_overlap"
            }
        ),
        "near_duplicate_candidates": len({r.eval_id for r in all_rows if r.candidate_type == "near_duplicate_text"}),
        "confirmed_eval_overlap": len(by_cls["confirmed_eval_overlap"]),
        "benign_train_split_overlap": len(by_cls["benign_train_split_overlap"]),
        "suspicious_near_duplicate": len(by_cls["suspicious_near_duplicate"]),
        "uncertain_manual_review": len(by_cls["uncertain_manual_review"]),
    }


def build_summary_md(
    stats: list[dict[str, Any]],
    warnings_out: list[str],
    field_notes: dict[str, Any],
    rebuttal_text: str,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Split-Aware Contamination Check Summary",
        "",
        "## Definition",
        "",
        "We only count overlaps between ViRL39K and the exact evaluation splits used in our experiments as evaluation contamination. Overlaps with official training splits are reported separately and are not counted as leakage.",
        "",
        "## Run configuration",
        "",
        f"- train-root: `{args.train_root}`",
        f"- eval-root: `{args.eval_root}`",
        f"- config: `{args.config}`",
        f"- text-threshold: {args.text_threshold}",
        f"- phash-strong: {args.phash_strong_threshold}",
        f"- phash-weak: {args.phash_weak_threshold}",
        f"- topk: {args.topk}",
        f"- max-samples: {args.max_samples}",
        "",
        "## Field resolution notes",
        "",
        "```json",
        json.dumps(field_notes, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Per-benchmark statistics",
        "",
        "| Eval benchmark | Eval split | # eval samples | Exact ID overlap | Exact text+image overlap | Near-duplicate candidates | Confirmed eval overlap | Benign train-split overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total_confirmed = 0
    total_eval = 0
    for s in stats:
        total_confirmed += s["confirmed_eval_overlap"]
        total_eval += s["n_eval_samples"]
        lines.append(
            f"| {s['eval_benchmark']} | {s['eval_split']} | {s['n_eval_samples']} | {s['exact_id_overlap']} | "
            f"{s['exact_text_image_overlap']} | {s['near_duplicate_candidates']} | {s['confirmed_eval_overlap']} | "
            f"{s['benign_train_split_overlap']} |"
        )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"- Confirmed evaluation overlaps (unique eval IDs): **{total_confirmed}** across **{total_eval}** eval samples.",
            f"- Confirmed overlap rate (by eval samples, upper bound): **{total_confirmed / total_eval * 100:.2f}%**" if total_eval else "- No eval samples loaded.",
            "- Recommend re-evaluation on overlap-removed splits if confirmed overlap > 0 for high-gain benchmarks.",
            "- All `uncertain_manual_review` and `suspicious_near_duplicate` rows are exported for manual audit.",
            "",
            "## Rebuttal-ready text",
            "",
            rebuttal_text,
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings_out:
        lines.extend(f"- {w}" for w in warnings_out)
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def build_rebuttal_text(total_confirmed: int, total_eval: int, stats: list[dict[str, Any]]) -> str:
    if total_confirmed == 0:
        result = (
            "We found no confirmed overlaps between filtered ViRL39K and the evaluation splits used in our experiments "
            "under split-aware metadata, normalized text, near-duplicate retrieval, and image hashing checks."
        )
    else:
        rate = total_confirmed / total_eval * 100 if total_eval else 0
        hotspots = ", ".join(
            f"{s['eval_benchmark']} ({s['confirmed_eval_overlap']}/{s['n_eval_samples']})"
            for s in stats
            if s["confirmed_eval_overlap"] > 0
        )
        result = (
            f"We found {total_confirmed} confirmed overlaps with evaluation splits ({rate:.1f}% of eval samples; "
            f"hotspots: {hotspots}). Overlaps with official training splits were reported separately and not counted as leakage."
        )
    return (
        "We performed a split-aware contamination check between ViRL39K and the exact evaluation splits used in our "
        "experiments. We used source metadata when available, normalized text matching, near-duplicate text retrieval, "
        "and image SHA256/perceptual hashing. Overlaps with official training splits were reported separately and not "
        f"counted as evaluation leakage. {result}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Split-aware ViRL39K contamination check")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_splits.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/contamination_check"))
    parser.add_argument("--text-threshold", type=float, default=0.90)
    parser.add_argument("--phash-strong-threshold", type=int, default=5)
    parser.add_argument("--phash-weak-threshold", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke test: limit rows per split")
    parser.add_argument("--skip-images", action="store_true", help="Skip image SHA256/pHash (ID + text only)")
    parser.add_argument(
        "--skip-near-duplicate",
        action="store_true",
        help="Skip TF-IDF near-duplicate retrieval (much faster on full train)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip benchmarks already listed in checkpoints/progress.json",
    )
    parser.add_argument(
        "--image-log-interval",
        type=int,
        default=500,
        help="Log every N newly hashed images (default 500)",
    )
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Parallel workers for image SHA256/pHash (default: min(8, CPU count))",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.out)
    checkpoint = CheckpointManager(args.out, logger)
    checkpoint._save_progress("init")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    warnings_out: list[str] = []
    if not HAS_PIL:
        warnings_out.append("PIL not available; image_sha256/phash will be empty.")
        logger.warn("PIL not available; image hashing disabled.")
    if not HAS_SKLEARN:
        warnings_out.append("sklearn not available; near-duplicate uses difflib fallback.")
        logger.warn("sklearn not available; near-duplicate will use slow difflib fallback.")

    aliases = cfg.get("source_dataset_aliases") or {}
    checkpoint._save_progress("load_train")
    train_records = load_train_manifest(args.train_root, cfg.get("train", {}), aliases, args.max_samples, warnings_out)
    logger.info(f"Loaded {len(train_records)} train records from {args.train_root}")
    write_json_atomic(checkpoint.dir / "train_loaded.json", {"n_train": len(train_records), "train_root": str(args.train_root)})

    near_dup_index: TrainNearDupIndex | None = None
    if not args.skip_near_duplicate and HAS_SKLEARN and train_records:
        train_texts = [t.normalized_question_choices for t in train_records]
        if any(train_texts):
            checkpoint._save_progress("build_near_dup_index")
            logger.info(f"Building shared TF-IDF index on {len(train_texts)} train texts...")
            t0 = time.time()
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            X_train = vec.fit_transform(train_texts)
            near_dup_index = TrainNearDupIndex(train_records, vec, X_train)
            logger.info(f"Shared TF-IDF index ready in {time.time() - t0:.1f}s")

    if not args.skip_images:
        checkpoint._save_progress("hash_train_images")
        enrich_image_hashes(
            train_records,
            warnings_out,
            checkpoint.image_cache,
            checkpoint=checkpoint,
            logger=logger,
            log_interval=args.image_log_interval,
            label="train",
            workers=args.hash_workers,
        )
        logger.info("Train image hashing complete")

    if args.resume and checkpoint.progress.get("completed_benchmarks"):
        all_overlap = checkpoint.load_completed_rows()
        stats = checkpoint.load_completed_stats()
        logger.info(
            f"Resume: loaded {len(all_overlap)} overlap rows from "
            f"{len(checkpoint.progress['completed_benchmarks'])} completed benchmarks"
        )
    else:
        all_overlap = []
        stats = []

    field_notes = {
        "train_fields": "question, answer, source, qid, image",
        "eval_fields": "ShareGPT messages/images via PAPO-Eval",
        "train_count": len(train_records),
    }

    eval_items = list((cfg.get("eval_splits") or {}).items())
    logger.info(f"Processing {len(eval_items)} evaluation benchmarks")

    for bi, (name, escfg) in enumerate(eval_items, start=1):
        if args.resume and checkpoint.is_benchmark_done(name):
            logger.info(f"[{bi}/{len(eval_items)}] SKIP {name} (already in checkpoint)")
            continue

        checkpoint._save_progress(f"benchmark:{name}:load")
        eval_records = load_eval_manifest(args.eval_root, name, escfg, args.max_samples, warnings_out)
        if not eval_records:
            logger.warn(f"[{bi}/{len(eval_items)}] {name}: no eval records, skipping")
            continue

        if not args.skip_images:
            checkpoint._save_progress(f"benchmark:{name}:hash_images")
            enrich_image_hashes(
                eval_records,
                warnings_out,
                checkpoint.image_cache,
                checkpoint=checkpoint,
                logger=logger,
                log_interval=min(args.image_log_interval, 100),
                label=f"eval:{name}",
                workers=args.hash_workers,
            )

        checkpoint._save_progress(f"benchmark:{name}:check")
        logger.info(f"[{bi}/{len(eval_items)}] Checking {name}: {len(eval_records)} eval samples")
        t0 = time.time()
        rows = run_checks(
            train_records,
            eval_records,
            name,
            escfg,
            text_threshold=args.text_threshold,
            phash_strong=args.phash_strong_threshold,
            phash_weak=args.phash_weak_threshold,
            topk=args.topk,
            warnings_out=warnings_out,
            skip_near_duplicate=args.skip_near_duplicate,
            logger=logger,
            near_dup_index=near_dup_index,
        )
        stat = aggregate_stats(name, escfg, eval_records, rows)
        logger.info(
            f"[{bi}/{len(eval_items)}] {name} done in {time.time() - t0:.1f}s — "
            f"confirmed={stat['confirmed_eval_overlap']}, "
            f"exact_id={stat['exact_id_overlap']}, "
            f"suspicious={stat['suspicious_near_duplicate']}"
        )

        all_overlap.extend(rows)
        stats.append(stat)
        field_notes[name] = {
            "eval_file": escfg.get("path"),
            "split": escfg.get("split"),
            "n_eval": len(eval_records),
            "test_id_parquet": escfg.get("test_id_parquet"),
        }

        checkpoint.save_benchmark_artifact(name, rows, stat)
        checkpoint.mark_benchmark_done(name)
        flush_aggregate_outputs(args.out, all_overlap)
        checkpoint.save_partial_summary(args.out, stats, all_overlap, warnings_out, len(train_records))
        logger.info(f"Checkpoint saved for {name} -> {checkpoint.dir / 'benchmarks' / (name + '_overlap.csv')}")

    checkpoint._save_progress("finalize")

    confirmed = [r for r in all_overlap if r.classification == "confirmed_eval_overlap"]
    suspicious = [r for r in all_overlap if r.classification == "suspicious_near_duplicate"]
    benign = [r for r in all_overlap if r.classification == "benign_train_split_overlap"]
    uncertain = [r for r in all_overlap if r.classification == "uncertain_manual_review"]
    candidates = [r for r in all_overlap if r.classification != "benign_train_split_overlap"]

    write_csv(args.out / "overlap_candidates.csv", candidates)
    write_csv(args.out / "confirmed_eval_overlap.csv", confirmed)
    write_csv(args.out / "near_duplicate_candidates.csv", suspicious)
    write_csv(args.out / "image_overlap_candidates.csv", [r for r in all_overlap if r.candidate_type in {"image_sha256", "phash_overlap"}])

    total_confirmed_ids = len({r.eval_id for r in confirmed})
    total_eval = sum(s["n_eval_samples"] for s in stats)
    rebuttal = build_rebuttal_text(total_confirmed_ids, total_eval, stats)

    summary = {
        "train_root": str(args.train_root),
        "eval_root": str(args.eval_root),
        "config": str(args.config),
        "thresholds": {
            "text": args.text_threshold,
            "phash_strong": args.phash_strong_threshold,
            "phash_weak": args.phash_weak_threshold,
            "topk": args.topk,
        },
        "train_samples": len(train_records),
        "per_benchmark": stats,
        "totals": {
            "eval_samples": total_eval,
            "confirmed_eval_overlap_unique_eval_ids": total_confirmed_ids,
            "overlap_candidates": len(candidates),
            "suspicious_near_duplicate": len(suspicious),
            "benign_train_split_overlap": len(benign),
            "uncertain_manual_review": len(uncertain),
        },
        "warnings": warnings_out,
        "rebuttal_text": rebuttal,
    }
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = build_summary_md(stats, warnings_out, field_notes, rebuttal, args)
    (args.out / "summary.md").write_text(md, encoding="utf-8")

    if confirmed:
        filtered = defaultdict(list)
        for r in confirmed:
            filtered[r.eval_dataset].append(r.eval_id)
        with open(args.out / "filtered_eval_ids.json", "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

    logger.info(f"Final summary: {args.out / 'summary.md'}")
    logger.info(f"Confirmed eval overlaps (unique eval IDs): {total_confirmed_ids}/{total_eval}")
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
