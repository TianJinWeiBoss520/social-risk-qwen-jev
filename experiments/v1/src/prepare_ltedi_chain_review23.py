#!/usr/bin/env python3
"""Offline diagnostic cards for validation-chain errors and disagreements.

No training, API calls, key access, serving, or final-test reads. Export is a
new private, self-contained HTML report; raw data, caches and labels remain
unchanged. These validation cases must NEVER become SFT/DPO training records.
Only export needs Pillow; --plan-only uses Python standard-library helpers.
"""

import argparse
import base64
import hashlib
import html
import io
import json
from collections import Counter
from pathlib import Path

import evaluate_ltedi_qwen_base as ev
import prepare_ltedi_jev_pilot as prep
import expand_ltedi_evidence_val172 as expand
import run_ltedi_jev_text_vs_multimodal172 as contrast


VERSION = "ltedi_validation_chain_review23_v1"
OUT_NAME = "ltedi_chain_review23_v1"
THRESHOLD = .40
ORDER = ("QWEN_CORRECT_JEV_WRONG", "BOTH_WRONG", "QWEN_WRONG_JEV_CORRECT")
TITLE = {
    ORDER[0]: "Qwen标签正确，Jev错误：优先检查证据及决策政策",
    ORDER[1]: "Qwen标签、Jev都错误：检查感知和语义理解",
    ORDER[2]: "Qwen标签错误，Jev正确：保护这部分已有收益",
}


def load_plan(root):
    root = root.resolve()
    out = root / "reports" / OUT_NAME
    if out.exists():
        raise ValueError("OUTPUT_EXISTS_STOP: preserve the prior diagnostic report")
    local = root / "outputs/ltedi_evidence_val172_v1"
    cloud = root / "outputs/jev_lora_remaining152_cloud_v1"
    if (local / ".local_worker.lock").exists():
        raise ValueError("LOCAL_EVIDENCE_WRITER_EXISTS_STOP")
    old_plan = prep.read_json(cloud / "plan.json")
    old = prep.read_json(cloud / "predictions_LOCAL_ONLY.json")
    report = prep.read_json(cloud / "report.json")
    if (old.get("metadata") != old_plan
            or old_plan.get("version") != contrast.previous.VERSION
            or old_plan.get("requested_model") != contrast.PINNED_MODEL
            or old_plan.get("test_read") is not False
            or old_plan.get("train_read") is not False
            or report.get("complete") is not True
            or report.get("valid_results") != 172
            or report.get("validation_n") != 172
            or report.get("test_read") is not False):
        raise ValueError("UNCHANGED_COMPLETED_VALIDATION_CACHE_REQUIRED")
    contrast.verify_sources(old_plan["source_sha256"])
    data = root / "data/processed/ltedi_grouped_v1"
    manifest = data / "val.jsonl"
    if old_plan["source_sha256"].get(str(manifest.resolve())) != ev.digest(manifest):
        raise ValueError("VALIDATION_MANIFEST_MISMATCH")
    rows = ev.load_validation(data)
    local_plan = prep.read_json(local / "plan.json")
    qwen = prep.read_json(local / "lora_evidence/predictions.json")["items"]
    jev = old["items"]
    if (len(rows) != 172 or set(qwen) != set(jev) or set(jev) != {r["id"] for r in rows}
            or local_plan.get("modality") != "image-text"
            or local_plan.get("validation_ids_LOCAL_ONLY") != [r["id"] for r in rows]):
        raise ValueError("VALIDATION_COVERAGE_OR_MULTIMODAL_PROVENANCE_MISMATCH")
    truth, half_scores = [], []
    counts = Counter()
    selected = []
    for row in rows:
        record = qwen[row["id"]]
        expand.check_record(record)
        if record["status"] != "ok":
            raise ValueError("UNRESOLVED_QWEN_RESULT")
        contrast.check_result(jev[row["id"]])
        p = jev[row["id"]]["violation_probability"]
        truth.append(row["label"])
        half_scores.append(int(p >= .5))
        q_ok = record["decision"]["label"] == row["label"]
        j_ok = int(p >= THRESHOLD) == row["label"]
        category = "BOTH_CORRECT" if q_ok and j_ok else ORDER[0] if q_ok else ORDER[2] if j_ok else ORDER[1]
        counts[category] += 1
        if category != "BOTH_CORRECT":
            selected.append({"id": row["id"], "group_id": row["group_id"], "source_split": "val",
                "eligible_for_training": False, "image": row["image"], "text": row["text"],
                "source_label": row["label"], "qwen": record["decision"],
                "jev_probability": p, "jev_label_t040": int(p >= THRESHOLD),
                "jev_error": None if j_ok else "FN" if row["label"] else "FP", "category": category})
    if ev.classification_metrics(truth, half_scores) != report["jev_classification_valid_only"]:
        raise ValueError("CACHED_METRICS_DO_NOT_REPRODUCE")
    selected.sort(key=lambda r: (ORDER.index(r["category"]), r["id"]))
    source_files = [manifest, cloud / "plan.json", cloud / "predictions_LOCAL_ONLY.json", cloud / "report.json",
        local / "plan.json", local / "lora_evidence/predictions.json", Path(__file__), Path(contrast.__file__)]
    source_hashes = {**old_plan["source_sha256"], **{str(f.resolve()): ev.digest(f) for f in source_files}}
    metadata = {"version": VERSION, "validation_n": len(rows), "selected_n": len(selected),
        "category_counts": dict(counts), "jev_threshold": THRESHOLD,
        "jev_errors": dict(Counter(r["jev_error"] for r in selected if r["jev_error"])),
        "source_sha256": source_hashes, "api_calls": 0, "gpu_used": False, "test_read": False,
        "train_read": False, "training_records_created": 0,
        "purpose": "VALIDATION_DIAGNOSIS_ONLY_NOT_TRAINING_DATA",
        "warning": "ID prefix train: denotes original source, NOT membership of the prepared training split. All exported rows belong to val.",
        "reason_for_selection": "All final-chain errors plus Qwen-wrong/Jev-correct controls; no automatic fault attribution."}
    return out, metadata, rows, selected, local_plan


def jpeg_data(path):
    from PIL import Image, ImageOps
    with Image.open(path) as picture:
        if getattr(picture, "n_frames", 1) != 1:
            raise ValueError("MULTIFRAME_REVIEW_REQUIRED")
        picture.load()
        picture = ImageOps.exif_transpose(picture).convert("RGB")
        picture.thumbnail((1200, 1200))
        stream = io.BytesIO()
        picture.save(stream, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


SCRIPT = """document.getElementById('save').addEventListener('click',()=>{
 const records=Array.from(document.querySelectorAll('article')).map(card=>({
  id_LOCAL_ONLY:card.dataset.id,source_split:'val',eligible_for_training:false,
  diagnosis:card.querySelector('select').value,note:card.querySelector('textarea').value
 }));
 const result={purpose:'VALIDATION_DIAGNOSIS_ONLY_NOT_TRAINING_DATA',records};
 const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='validation_review_NOT_TRAINING.json';
 link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});"""


def render(selected, images, metadata):
    escape = html.escape
    script_sha = base64.b64encode(hashlib.sha256(SCRIPT.encode("utf-8")).digest()).decode("ascii")
    csp = f"default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'sha256-{script_sha}'; connect-src 'none'; base-uri 'none'; form-action 'none'"
    cards = []
    choices = ("待核实", "证据错误：文字/OCR", "证据错误：图像事实", "证据错误：对象/立场/反讽",
               "证据错误：无依据推断", "证据基本正确：Jev政策/决策待优化", "标签或语境有争议", "已正确：保留收益")
    for index, row in enumerate(selected, 1):
        decision = escape(json.dumps(row["qwen"], ensure_ascii=False, indent=2))
        options = "".join(f"<option>{escape(choice)}</option>" for choice in choices)
        cards.append(f'''<article data-id="{escape(row['id'], quote=True)}">
<h2>{index}. {escape(row['id'])} · {escape(TITLE[row['category']])}</h2>
<p>原始标签：{row['source_label']} · Qwen标签：{row['qwen']['label']} · Jev：{row['jev_probability']:.4f}
 · Jev类别（阈值0.40）：{row['jev_label_t040']} · {escape(row['jev_error'] or 'Jev正确')}</p>
<div class="content"><img src="{images[row['id']]}" alt="待核实的原始图像（报告缩略图）">
<div><h3>原始转录</h3><pre>{escape(row['text'])}</pre><h3>Qwen生成的判断与证据（不是事实标注）</h3><pre>{decision}</pre>
<p>label仅用于本地诊断，不会发送给Jev。</p>
<label>人工诊断 <select>{options}</select></label><br>
<label>依据/备注 <textarea placeholder="记录原图和文本中的依据；不要为了匹配标签而补造证据。"></textarea></label></div></div>
</article>''')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{escape(csp, quote=True)}">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen＋Jev 验证案例诊断</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;background:#f5f6f8;color:#17202a}}header,article{{background:white;padding:20px;margin:18px 0;border-radius:10px}}h2{{font-size:18px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:12px}}.content{{display:grid;grid-template-columns:minmax(250px,1fr) minmax(320px,1fr);gap:24px}}img{{width:100%;height:auto;max-height:900px;object-fit:contain;align-self:start}}textarea{{display:block;width:95%;min-height:80px;margin-top:8px}}select,button{{padding:8px}}.warning{{background:#fff3cd;padding:12px}}@media(max-width:800px){{.content{{display:block}}}}</style></head>
<body><header><h1>Qwen＋Jev：验证案例诊断</h1>
<p class="warning">这不是训练数据！所有卡片来自172条验证集，包括ID以train:开头的案例。不得把这些图像、答案或审核记录加入SFT/DPO训练。已评估的170条测试集未读取。</p>
<p>全量标签交叉计数：{escape(json.dumps(metadata['category_counts'], ensure_ascii=False))}</p>
<p>优先看前4条分歧，再看共同错误，最后检查Jev已经纠正的案例。标签一致性不能证明证据真实；疑似标注争议可保留，不强行改标签。</p>
<p>内容可能含冒犯性表达。报告包含本地图像与原始转录，请仅在个人电脑打开，不发布到公共网站。无外部资源、API或网络请求。</p>
<button id="save">下载本地诊断记录（不是训练数据）</button>
<p>输入内容不会自动保存；关闭前请下载记录。记录用于错误归因，不生成chosen/rejected，也不修改源标签。</p></header>
{''.join(cards)}<script>{SCRIPT}</script></body></html>'''


def export(out, metadata, rows, selected, local_plan):
    # Verify the original images match those used for the cached perception.
    expected = local_plan.get("source_validation_image_bytes_sha256")
    if not expected:
        raise ValueError("ORIGINAL_VALIDATION_IMAGE_HASH_MISSING")
    image_hashes = [(r["id"], ev.digest(Path(r["image"]))) for r in rows]
    actual = hashlib.sha256(json.dumps(image_hashes, sort_keys=True).encode()).hexdigest()
    if expected != actual:
        raise ValueError("VALIDATION_IMAGES_CHANGED_STOP")
    images = {row["id"]: jpeg_data(Path(row["image"])) for row in selected}
    contrast.verify_sources(metadata["source_sha256"])
    document = render(selected, images, metadata)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    with (out / "review_PRIVATE.html").open("x", encoding="utf-8") as stream:
        stream.write(document)
    ev.atomic_json(out / "manifest_DIAGNOSIS_ONLY.json", {"metadata": metadata, "items": selected})
    print(f"REVIEW_READY={out / 'review_PRIVATE.html'} CASES={len(selected)}", flush=True)
    print("NEW_LOCAL_REPORT_ONLY; API_CALLS=0 GPU=False TRAINING_RECORDS=0 TEST_READ=False SOURCE_FILES_UNCHANGED", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ev.DEFAULT_ROOT)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    out, metadata, rows, selected, local_plan = load_plan(args.project_root)
    print(f"REVIEW_PLAN VAL={len(rows)} SELECTED={len(selected)} THRESHOLD=0.40", flush=True)
    print("CATEGORY_COUNTS=" + prep.canonical(metadata["category_counts"]), flush=True)
    print("JEV_ERRORS=" + prep.canonical(metadata["jev_errors"]), flush=True)
    if args.plan_only:
        print("PLAN_ONLY API_CALLS=0 KEY_READ=False GPU=False IMAGE_DECODE=False OUTPUT_WRITTEN=False TEST_READ=False", flush=True)
        return 0
    export(out, metadata, rows, selected, local_plan)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Local diagnostic errors only; no key or network client is accessed.
        print(f"REVIEW_STOP ERROR_TYPE={type(exc).__name__} REASON={exc}; preserve existing files", flush=True)
        raise SystemExit(2)
