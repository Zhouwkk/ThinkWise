import json
import importlib.util
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

p = hf_hub_download("FanqingM/MMK12", "data/test-00000-of-00001.parquet", repo_type="dataset")
df = pq.read_table(p).to_pandas()
test_ids = set(df["id"].astype(str))
print("MMK12 test n=", len(test_ids))

with open("/data/zhouwenkang/FAST/train_data/ViRL39K/virl39k_filtered.json") as f:
    tr = json.load(f)
train_id_suffixes = set()
for r in tr:
    qid = r.get("qid", "")
    if qid.startswith("MMK12-"):
        train_id_suffixes.add(qid[len("MMK12-") :])

id_overlap = test_ids & train_id_suffixes
print("ID-based test-in-train:", len(id_overlap), "/", len(test_ids))

spec = importlib.util.spec_from_file_location(
    "e4", "/data/zhouwenkang/FAST/experiments/e4/e4_contamination_check.py"
)
e4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e4)
with open("/data/zhouwenkang/PAPO-Eval/data/papo/PAPO_MMK12.json") as f:
    ev = json.load(f)
train_qs = {e4.normalize_question(r["question"]) for r in tr}
text_hits = sum(
    1
    for item in ev
    if e4.normalize_question(next(m["content"] for m in item["messages"] if m["role"] == "user"))
    in train_qs
)
print("Text-based overlap (old):", text_hits, "/", len(ev))
