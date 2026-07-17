#!/usr/bin/env python3
"""Content-based contamination analysis: ViRL39K (train) vs PAPO-Eval (test).

Goal
----
Answer the reviewer's question "does the training set contain test-set data?"
using only *trustworthy, content-based* evidence, so the resulting numbers are
safe to put in a rebuttal.

Signals
-------
STRICT contamination (primary metric for rebuttal):
  Same problem instance in train and eval: normalized question(+choices) are
  equivalent (exact match OR char similarity >= --question-sim-threshold) AND
  the *semantic* correct answer matches. For multiple-choice items, option
  letters (B vs D) are resolved to option *content* before comparing, so
  shuffled choices do not false-negative; letter-only match is NOT used.

SHARED ASSET (reported separately, NOT strict contamination):
  Identical image SHA256 but the pair fails the strict Q+A test — typically
  a reused diagram with a different question and/or answer.

REVIEW candidates:
  near-duplicate question text (TF-IDF) or identical perceptual hash (aHash)
  without passing strict Q+A; also pairs with matching question but different
  answers (possible relabeling or template reuse).

Deliberately NOT used
---------------------
The alias-based numeric-ID matching from the older `check_contamination.py`.
That logic mapped a ViRL39K `source` to a benchmark via aliases (e.g.
`MMMath -> MathVision`), stripped the qid to a bare number, and matched it
against benchmark test-row indices. On MathVision this produced ~1933 *false*
matches: ViRL39K `MMMath-1` was "matched" to MathVision test row #1 only
because both reduce to the integer "1", even though the question, image, and
answer are completely different. This script avoids ID heuristics entirely and
only compares actual content.

Notes
-----
* Eval JSONs are ShareGPT-style: {"messages": [...], "images": [...]} with no
  id field. We parse the user turn as the question and the assistant turn as
  the answer.
* Image hashes are cached on disk; pass --reuse-cache to seed from a previous
  run's cache (keyed by resolved absolute path).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# 10 evaluation benchmarks (Table 2 / contamination check). (display, file, split)
BENCHMARKS: list[tuple[str, str, str]] = [
    ("MathVista", "AI4Math_MathVista.json", "testmini"),
    ("MathVerse", "AI4Math_MathVerse.json", "test"),
    ("MathVerse_v", "AI4Math_MathVerse_vision_dependent.json", "test"),
    ("Counting", "BUAADreamer_clevr_count_70k.json", "test"),
    ("MMMU-Pro", "MMMU_MMMU_Pro.json", "test"),
    ("MathVision", "MathLLMs_MathVision.json", "test"),
    ("MMK12", "PAPO_MMK12.json", "test"),
    ("We-Math", "We_Math.json", "test"),
    ("Geo3k", "hiyouga_geometry3k.json", "test"),
    ("LogicVista", "lscpku_LogicVista.json", "test"),
]

INSTRUCTION_SUFFIX = re.compile(
    r"\n?\s*You first think through the reasoning process.*?\\boxed\{\}\.\s*$",
    re.DOTALL | re.IGNORECASE,
)
IMAGE_TOKEN = re.compile(r"<image\s*\d*>", re.IGNORECASE)
CHOICES_BLOCK = re.compile(r"Choices:\s*(.+?)(?:\n\n|\Z)", re.DOTALL | re.IGNORECASE)
CHOICES_BRACKET = re.compile(r"\[([^\]]+)\]")
TRAIN_WRAPPER_PREFIX = re.compile(
    r"^the below problem is with the following images:\s*",
    re.IGNORECASE,
)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = TRAIN_WRAPPER_PREFIX.sub("", text)
    text = IMAGE_TOKEN.sub(" ", text)
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


def normalize_answer_value(text: str) -> str:
    """Normalize a semantic answer value (not an option letter)."""
    a = strip_boxed(text or "")
    a = a.strip().lower()
    a = re.sub(r"\s+", "", a)
    a = re.sub(r"\$+", "", a)
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\(?:approx|sim|simeq|eq)?", "", a)
    return a


MC_LETTER = re.compile(r"^[\(\[]?([a-d])[\)\].]?$", re.IGNORECASE)
MC_LINE = re.compile(r"^([A-Da-d])[\.\)]\s*(.+)$")


def parse_choice_map(question: str) -> dict[str, str]:
    """Map option letter -> normalized option content from the question text."""
    choices: dict[str, str] = {}
    for line in question.splitlines():
        line = line.strip()
        m = MC_LINE.match(line)
        if m:
            letter = m.group(1).upper()
            choices[letter] = normalize_answer_value(m.group(2))
    if choices:
        return choices
    # Inline "A: ... B: ..." fallback
    block = CHOICES_BLOCK.search(question)
    if block:
        for part in re.split(r"\s+(?=[A-D]\.)", block.group(1)):
            m = re.match(r"^([A-D])\.\s*(.+)$", part.strip(), re.IGNORECASE)
            if m:
                choices[m.group(1).upper()] = normalize_answer_value(m.group(2))
    return choices


def resolve_semantic_answer(question: str, raw_answer: str) -> str:
    """Resolve MC letter to option content; otherwise return normalized value.

    Contamination is about whether the *correct solution* leaked (e.g. 9), not
    whether the same option letter (B vs D) was used when choice order differs.
    """
    raw = strip_boxed(raw_answer or "")
    if not raw:
        return ""
    letter_m = MC_LETTER.match(raw.strip())
    if letter_m:
        letter = letter_m.group(1).upper()
        choices = parse_choice_map(question)
        if letter in choices and choices[letter]:
            return choices[letter]
        return ""  # letter answer but cannot resolve — do not compare as letter
    return normalize_answer_value(raw)


def answers_match(er: "Record", tr: "Record") -> bool:
    """True if train and eval share the same semantic correct answer."""
    sa = resolve_semantic_answer(er.question, er.answer)
    sb = resolve_semantic_answer(tr.question, tr.answer)
    if not sa or not sb:
        return False
    return sa == sb


def question_similarity(er: "Record", tr: "Record") -> float:
    if er.norm_qc and er.norm_qc == tr.norm_qc:
        return 1.0
    if not er.norm_qc or not tr.norm_qc:
        return 0.0
    return SequenceMatcher(None, er.norm_qc, tr.norm_qc).ratio()


def is_strict_same_problem(
    er: "Record", tr: "Record", *, question_sim_threshold: float
) -> tuple[bool, float, bool]:
    """Return (is_strict_leak, q_sim, ans_match).

    Strict if the full instance matches (norm Q+C+A) OR (similar question AND same answer).
    """
    ans_ok = answers_match(er, tr)
    if er.norm_qca and er.norm_qca == tr.norm_qca and ans_ok:
        return True, 1.0, ans_ok
    q_sim = question_similarity(er, tr)
    same_problem = q_sim >= question_sim_threshold
    return same_problem and ans_ok, q_sim, ans_ok


# ---------------------------------------------------------------------------
# Image hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def average_phash(path: Path) -> str:
    with Image.open(path) as im:
        im = im.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(im.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p >= avg else "0" for p in pixels)


def _hash_worker(path_str: str) -> tuple[str, str, str]:
    p = Path(path_str)
    if not p.is_file():
        return path_str, "", ""
    try:
        sha = sha256_file(p)
        ph = average_phash(p) if HAS_PIL else ""
    except Exception:
        return path_str, "", ""
    return path_str, sha, ph


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Record:
    rec_id: str
    question: str
    answer: str
    norm_qc: str  # normalized question + choices
    norm_qca: str  # normalized question + choices + answer (strongest same-instance test)
    source: str = ""
    image_paths: list[str] = field(default_factory=list)
    image_shas: list[str] = field(default_factory=list)
    image_phashes: list[str] = field(default_factory=list)


def resolve_images(raw: Any, root: Path) -> list[str]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for it in items:
        if not isinstance(it, str) or not it:
            continue
        rel = it[2:] if it.startswith("./") else it
        p = rel if Path(rel).is_absolute() else str((root / rel).resolve())
        out.append(p)
    return out


def load_train(path: Path, image_root: Path) -> list[Record]:
    data = json.loads(path.read_text(encoding="utf-8"))
    recs: list[Record] = []
    for i, row in enumerate(data):
        q = str(row.get("question", ""))
        choices = extract_choices(q)
        ans = strip_boxed(str(row.get("answer", "")))
        nqc = normalize_text(f"{q} {choices}".strip())
        sem_ans = resolve_semantic_answer(q, ans)
        recs.append(
            Record(
                rec_id=str(row.get("qid", f"train_{i}")),
                question=q,
                answer=ans,
                norm_qc=nqc,
                norm_qca=normalize_text(f"{q} {choices} {sem_ans}".strip()),
                source=str(row.get("source", "")),
                image_paths=resolve_images(row.get("image"), image_root),
            )
        )
    return recs


def load_eval(path: Path, eval_root: Path) -> list[Record]:
    data = json.loads(path.read_text(encoding="utf-8"))
    recs: list[Record] = []
    for i, row in enumerate(data):
        q, ans = "", ""
        for msg in row.get("messages") or []:
            content = (msg.get("content") or "").strip()
            if msg.get("role") == "user":
                q = content
            elif msg.get("role") == "assistant":
                ans = content
        choices = extract_choices(q)
        ans_clean = strip_boxed(ans)
        nqc = normalize_text(f"{q} {choices}".strip())
        sem_ans = resolve_semantic_answer(q, ans_clean)
        recs.append(
            Record(
                rec_id=f"{path.stem}_{i}",
                question=q,
                answer=ans_clean,
                norm_qc=nqc,
                norm_qca=normalize_text(f"{q} {choices} {sem_ans}".strip()),
                image_paths=resolve_images(row.get("images") or row.get("image"), eval_root),
            )
        )
    return recs


def enrich_image_hashes(
    records: list[Record],
    cache: dict[str, tuple[str, str]],
    cache_fp,
    workers: int,
    label: str,
    missing: list[str],
) -> None:
    pending = sorted({p for r in records for p in r.image_paths if p not in cache})
    if pending:
        log(f"{label}: hashing {len(pending)} new images ({len(cache)} cached, workers={workers})")
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for path_str, sha, ph in pool.map(_hash_worker, pending, chunksize=64):
                cache[path_str] = (sha, ph)
                cache_fp.write(json.dumps({"path": path_str, "sha256": sha, "phash": ph}) + "\n")
                if not sha:
                    missing.append(path_str)
                done += 1
                if done % 5000 == 0:
                    log(f"{label}: {done}/{len(pending)} hashed ({done / (time.time() - t0):.0f}/s)")
        cache_fp.flush()
    for r in records:
        for p in r.image_paths:
            sha, ph = cache.get(p, ("", ""))
            r.image_shas.append(sha)
            r.image_phashes.append(ph)


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    tier: str  # strict_leakage | shared_image_asset | same_q_diff_answer | review_*
    link: str  # exact_qc | image_sha256 | near_dup_text | image_phash_exact
    benchmark: str
    split: str
    eval_id: str
    train_id: str
    train_source: str
    q_similarity: float
    answer_match: bool
    eval_question: str
    train_question: str
    eval_answer: str
    train_answer: str


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-json", type=Path, default=Path("/data/zhouwenkang/FAST/train_data/ViRL39K/virl39k_train.json"))
    ap.add_argument("--train-image-root", type=Path, default=Path("/data/zhouwenkang/FAST/train_data/ViRL39K"))
    ap.add_argument("--eval-root", type=Path, default=Path("/data/zhouwenkang/PAPO-Eval"))
    ap.add_argument("--eval-subdir", type=str, default="data/papo")
    ap.add_argument("--out", type=Path, default=Path("/data/zhouwenkang/FAST/outputs/contamination_analysis"))
    ap.add_argument("--text-threshold", type=float, default=0.90, help="TF-IDF cosine threshold for near-dup text candidates")
    ap.add_argument(
        "--question-sim-threshold",
        type=float,
        default=0.95,
        help="Min question(+choices) similarity for strict leakage when not exact match",
    )
    ap.add_argument("--topk", type=int, default=3, help="top-k train neighbours per eval item for near-dup text")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse-cache", type=Path, default=Path("/data/zhouwenkang/FAST/outputs/contamination_check/checkpoints/image_hash_cache.jsonl"), help="seed image-hash cache from a previous run")
    ap.add_argument("--max-samples", type=int, default=None, help="debug: cap rows per file")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if not HAS_PIL and not args.skip_images:
        log("WARN: PIL unavailable; image signals disabled.")
        args.skip_images = True

    # ---- load train ----
    log(f"Loading train: {args.train_json}")
    train = load_train(args.train_json, args.train_image_root)
    if args.max_samples:
        train = train[: args.max_samples]
    log(f"Loaded {len(train)} train records")

    # ---- image hash cache ----
    cache: dict[str, tuple[str, str]] = {}
    missing: list[str] = []
    if not args.skip_images:
        if args.reuse_cache and args.reuse_cache.is_file():
            with open(args.reuse_cache, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        cache[row["path"]] = (row.get("sha256", ""), row.get("phash", ""))
            log(f"Seeded {len(cache)} hashes from {args.reuse_cache}")

    cache_path = args.out / "image_hash_cache.jsonl"
    cache_fp = open(cache_path, "a", encoding="utf-8") if not args.skip_images else None

    if not args.skip_images:
        enrich_image_hashes(train, cache, cache_fp, args.workers, "train", missing)

    # ---- build train indices ----
    train_by_text: dict[str, list[int]] = defaultdict(list)
    train_by_qca: dict[str, list[int]] = defaultdict(list)
    train_by_sha: dict[str, list[int]] = defaultdict(list)
    train_by_phash: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(train):
        if r.norm_qc:
            train_by_text[r.norm_qc].append(i)
        if r.norm_qca:
            train_by_qca[r.norm_qca].append(i)
        for sha in r.image_shas:
            if sha:
                train_by_sha[sha].append(i)
        for ph in r.image_phashes:
            if ph:
                train_by_phash[ph].append(i)

    # ---- TF-IDF index over train text (shared) ----
    log("Building TF-IDF char index over train text...")
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    train_texts = [r.norm_qc for r in train]
    X_train = vec.fit_transform(train_texts)
    log(f"TF-IDF index ready in {time.time() - t0:.1f}s ({X_train.shape[1]} features)")

    strict_hits: list[Hit] = []
    shared_image_hits: list[Hit] = []
    same_q_diff_ans_hits: list[Hit] = []
    review_hits: list[Hit] = []
    per_bench: list[dict[str, Any]] = []

    def add_hit(
        hits: list[Hit],
        *,
        tier: str,
        link: str,
        name: str,
        split: str,
        er: Record,
        tr: Record,
        q_sim: float,
        ans_ok: bool,
    ) -> None:
        hits.append(
            Hit(
                tier=tier,
                link=link,
                benchmark=name,
                split=split,
                eval_id=er.rec_id,
                train_id=tr.rec_id,
                train_source=tr.source,
                q_similarity=q_sim,
                answer_match=ans_ok,
                eval_question=er.question[:400],
                train_question=tr.question[:400],
                eval_answer=er.answer,
                train_answer=tr.answer,
            )
        )

    for bi, (name, fname, split) in enumerate(BENCHMARKS, 1):
        eval_path = args.eval_root / args.eval_subdir / fname
        if not eval_path.is_file():
            log(f"[{bi}/10] SKIP {name}: missing {eval_path}")
            continue
        ev = load_eval(eval_path, args.eval_root)
        if args.max_samples:
            ev = ev[: args.max_samples]
        if not args.skip_images:
            enrich_image_hashes(ev, cache, cache_fp, args.workers, f"eval:{name}", missing)

        strict_ids: set[str] = set()
        shared_img_ids: set[str] = set()
        diff_ans_ids: set[str] = set()
        review_ids: set[str] = set()
        strict_sources: Counter = Counter()

        for er in ev:
            cand: set[int] = set()
            if er.norm_qc:
                cand.update(train_by_text.get(er.norm_qc, []))
            if er.norm_qca:
                cand.update(train_by_qca.get(er.norm_qca, []))
            if not args.skip_images:
                for sha in er.image_shas:
                    if sha:
                        cand.update(train_by_sha.get(sha, []))

            strict_cands: list[tuple[float, int, str]] = []
            image_cands: list[tuple[float, int]] = []
            q_diff_ans_cands: list[tuple[float, int, str]] = []

            for ti in cand:
                tr = train[ti]
                strict, q_sim, ans_ok = is_strict_same_problem(
                    er, tr, question_sim_threshold=args.question_sim_threshold
                )
                linked_by_img = bool(
                    not args.skip_images
                    and er.image_shas
                    and tr.image_shas
                    and set(er.image_shas) & set(tr.image_shas)
                )
                linked_by_qc = bool(er.norm_qc and er.norm_qc == tr.norm_qc)

                if strict:
                    if er.norm_qca and er.norm_qca == tr.norm_qca:
                        link = "exact_qca"
                    elif linked_by_qc:
                        link = "exact_qc"
                    else:
                        link = "high_q_sim"
                    if linked_by_img:
                        link += "+image"
                    strict_cands.append((q_sim, ti, link))
                elif linked_by_img:
                    image_cands.append((q_sim, ti))
                elif (linked_by_qc or q_sim >= args.question_sim_threshold) and not ans_ok:
                    link = "exact_qc" if linked_by_qc else "high_q_sim"
                    q_diff_ans_cands.append((q_sim, ti, link))

            if strict_cands:
                q_sim, ti, link = max(strict_cands, key=lambda x: x[0])
                tr = train[ti]
                add_hit(
                    strict_hits,
                    tier="strict_leakage",
                    link=link,
                    name=name,
                    split=split,
                    er=er,
                    tr=tr,
                    q_sim=q_sim,
                    ans_ok=True,
                )
                strict_ids.add(er.rec_id)
                strict_sources[tr.source] += 1
            elif image_cands:
                q_sim, ti = max(image_cands, key=lambda x: x[0])
                tr = train[ti]
                add_hit(
                    shared_image_hits,
                    tier="shared_image_asset",
                    link="image_sha256",
                    name=name,
                    split=split,
                    er=er,
                    tr=tr,
                    q_sim=q_sim,
                    ans_ok=answers_match(er, tr),
                )
                shared_img_ids.add(er.rec_id)
            elif q_diff_ans_cands:
                q_sim, ti, link = max(q_diff_ans_cands, key=lambda x: x[0])
                tr = train[ti]
                add_hit(
                    same_q_diff_ans_hits,
                    tier="same_q_diff_answer",
                    link=link,
                    name=name,
                    split=split,
                    er=er,
                    tr=tr,
                    q_sim=q_sim,
                    ans_ok=False,
                )
                diff_ans_ids.add(er.rec_id)

        # Near-duplicate text (review) — only if not already strict
        eval_texts = [r.norm_qc for r in ev]
        if any(eval_texts):
            X_eval = vec.transform(eval_texts)
            sims = cosine_similarity(X_eval, X_train)
            for ei, er in enumerate(ev):
                if not er.norm_qc or er.rec_id in strict_ids:
                    continue
                scores = sims[ei]
                top = np.argpartition(scores, -args.topk)[-args.topk:]
                for ti in top:
                    s = float(scores[ti])
                    if s < args.text_threshold or s >= 0.999:
                        continue
                    tr = train[int(ti)]
                    _, q_sim, ans_ok = is_strict_same_problem(
                        er, tr, question_sim_threshold=args.question_sim_threshold
                    )
                    add_hit(
                        review_hits,
                        tier="review_near_dup_text",
                        link="near_dup_text",
                        name=name,
                        split=split,
                        er=er,
                        tr=tr,
                        q_sim=max(s, q_sim),
                        ans_ok=ans_ok,
                    )
                    review_ids.add(er.rec_id)
            del sims, X_eval

        # Near-duplicate image (aHash, review)
        if not args.skip_images:
            for er in ev:
                if er.rec_id in strict_ids or er.rec_id in shared_img_ids:
                    continue
                er_shaset = {s for s in er.image_shas if s}
                for ph in er.image_phashes:
                    if not ph:
                        continue
                    for ti in train_by_phash.get(ph, []):
                        tr = train[ti]
                        if er_shaset & {s for s in tr.image_shas if s}:
                            continue
                        add_hit(
                            review_hits,
                            tier="review_image_phash",
                            link="image_phash_exact",
                            name=name,
                            split=split,
                            er=er,
                            tr=tr,
                            q_sim=question_similarity(er, tr),
                            ans_ok=answers_match(er, tr),
                        )
                        review_ids.add(er.rec_id)

        review_only = review_ids - strict_ids - shared_img_ids - diff_ans_ids
        per_bench.append({
            "benchmark": name,
            "split": split,
            "n_eval": len(ev),
            "strict_leakage": len(strict_ids),
            "strict_rate": round(len(strict_ids) / len(ev), 4) if ev else 0.0,
            "shared_image_asset": len(shared_img_ids),
            "same_question_diff_answer": len(diff_ans_ids),
            "review_candidates": len(review_only),
            "top_train_sources_strict": dict(strict_sources.most_common(5)),
        })
        log(
            f"[{bi}/10] {name}: n={len(ev)} strict={len(strict_ids)} "
            f"shared_img={len(shared_img_ids)} same_q_diff_ans={len(diff_ans_ids)} review={len(review_only)}"
        )

    if cache_fp:
        cache_fp.close()

    # ---- write outputs ----
    def write_hits(path: Path, hits: list[Hit]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "tier",
                    "link",
                    "benchmark",
                    "split",
                    "eval_id",
                    "train_id",
                    "train_source",
                    "q_similarity",
                    "answer_match",
                    "eval_question",
                    "train_question",
                    "eval_answer",
                    "train_answer",
                ]
            )
            for h in hits:
                w.writerow(
                    [
                        h.tier,
                        h.link,
                        h.benchmark,
                        h.split,
                        h.eval_id,
                        h.train_id,
                        h.train_source,
                        f"{h.q_similarity:.4f}",
                        h.answer_match,
                        h.eval_question,
                        h.train_question,
                        h.eval_answer,
                        h.train_answer,
                    ]
                )

    write_hits(args.out / "strict_leakage.csv", strict_hits)
    write_hits(args.out / "shared_image_asset.csv", shared_image_hits)
    write_hits(args.out / "same_question_diff_answer.csv", same_q_diff_ans_hits)
    write_hits(args.out / "review_candidates.csv", review_hits)

    total_eval = sum(b["n_eval"] for b in per_bench)
    total_strict = sum(b["strict_leakage"] for b in per_bench)
    total_shared = sum(b["shared_image_asset"] for b in per_bench)
    total_diff = sum(b["same_question_diff_answer"] for b in per_bench)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_json": str(args.train_json),
        "n_train": len(train),
        "eval_root": str(args.eval_root),
        "text_threshold": args.text_threshold,
        "question_sim_threshold": args.question_sim_threshold,
        "method": (
            "STRICT leakage = same problem (norm question+choices match or q_sim >= threshold) "
            "AND semantic answer match (MC: letter -> option content). Shared-image-only = "
            "identical SHA256 but fails strict Q+semantic-A. No alias/ID matching."
        ),
        "per_benchmark": per_bench,
        "totals": {
            "n_eval": total_eval,
            "strict_leakage": total_strict,
            "strict_rate": round(total_strict / total_eval, 4) if total_eval else 0.0,
            "shared_image_asset": total_shared,
            "same_question_diff_answer": total_diff,
        },
        "n_missing_images": len(missing),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# ViRL39K vs PAPO-Eval Contamination Analysis (Q+A strict)",
        "",
        f"- Train: `{args.train_json}` ({len(train)} records)",
        f"- **Strict leakage** = similar question (exact norm QC or q_sim ≥ {args.question_sim_threshold}) **and** same semantic answer (MC letters resolved to option content, not letter-only).",
        "- **Shared image asset** = identical image SHA256 only (different task and/or label) — **not** counted as test contamination.",
        "- **Same question, different answer** = high question similarity but mismatched labels (manual review).",
        "",
        "| Benchmark | Split | # eval | Strict leak | Rate | Shared img only | Q same / A diff | Review |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in per_bench:
        lines.append(
            f"| {b['benchmark']} | {b['split']} | {b['n_eval']} | {b['strict_leakage']} | "
            f"{b['strict_rate']*100:.1f}% | {b['shared_image_asset']} | {b['same_question_diff_answer']} | "
            f"{b['review_candidates']} |"
        )
    lines += [
        f"| **Total** | | **{total_eval}** | **{total_strict}** | "
        f"**{(total_strict/total_eval*100 if total_eval else 0):.1f}%** | **{total_shared}** | **{total_diff}** | |",
        "",
        "## Notes",
        "",
        f"- Missing/unreadable images: {len(missing)}",
        "- Primary file for rebuttal: `strict_leakage.csv`.",
        "- Legacy broad metric (image OR text): see archived `confirmed_overlap.csv` from prior run if needed.",
    ]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    log(
        f"Done. Strict leakage {total_strict}/{total_eval}; "
        f"shared-image-only {total_shared}; Q-same-A-diff {total_diff}. "
        f"Summary: {args.out / 'summary.md'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
