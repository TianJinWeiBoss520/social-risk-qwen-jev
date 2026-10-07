#!/usr/bin/env python3
"""Private, CPU-only eligibility/duplicate review; never changes source data.

Place beside audit_ltedi_images.py and download_ltedi_images.py on AutoDL.
prepare writes a NEW manifest/database; serve binds ONLY 127.0.0.1.
Review decisions are NOT harm labels, training splits, or guarantees of safety.
"""

import argparse
import csv
import hashlib
import io
import json
import secrets
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageOps

from audit_ltedi_images import decode_record
from download_ltedi_images import COMMIT, DEFAULT_REPO, file_blob_sha1, read_specs


DEFAULT_REVIEW = "/root/autodl-tmp/social_risk_jev/data/review/ltedi_v1"
CONTENT_CHOICES = {"keep", "exclude_political", "exclude_uncertain"}
PAIR_CHOICES = {"same_content", "same_template", "different_content", "uncertain"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_image(repo, relative):
    path = repo / relative
    if Path(relative).is_absolute() or "\\" in relative or ".." in Path(relative).parts:
        raise ValueError("Unsafe image path")
    if path.is_symlink() or not path.resolve().is_relative_to(repo.resolve()):
        raise ValueError("Image escapes repository")
    if not path.is_file():
        raise ValueError("Image is not a regular file")
    return path


def initialize_database(path):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE decisions (
                kind TEXT NOT NULL, target TEXT NOT NULL, decision TEXT NOT NULL,
                note TEXT NOT NULL, updated_utc TEXT NOT NULL,
                PRIMARY KEY(kind, target)
            );
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                target TEXT NOT NULL, decision TEXT NOT NULL, note TEXT NOT NULL,
                updated_utc TEXT NOT NULL
            );
        """)


def prepare(repo, output):
    repo, output = repo.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError(f"REVIEW_OUTPUT_EXISTS_STOP: {output}; use status/serve, not prepare again")
    if output.is_relative_to(repo) or repo.is_relative_to(output):
        raise RuntimeError("Review output must be separate from the source repository")
    specs = read_specs(repo)  # Pinned local Git tree/CSV only; no fetch, upload, or model.
    annotations = {}
    for split in ("train", "dev"):
        with (repo / f"Datasets/{split}/{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                annotations[f"{split}:{row['image_name'].strip()}"] = row

    records, rows, failures = [], [], []
    for index, spec in enumerate(specs, 1):
        key = f"{spec.split}:{Path(spec.relative).name}"
        annotation = annotations[key]
        path = safe_image(repo, spec.relative)
        if file_blob_sha1(path) != spec.sha1:
            raise RuntimeError(f"SOURCE_BYTES_CHANGED_STOP: {key}")
        row = {
            "id": key, "original_split": spec.split, "image_relative": spec.relative,
            "text": annotation["transcriptions"], "source_label": annotation["labels"].strip(),
            "source_git_blob_sha1": spec.sha1,
        }
        try:
            record = decode_record(path, key, spec.split, row["source_label"])
            records.append(record)
            row.update(size=record.size, file_sha256=record.file_sha256,
                       pixel_sha256=record.pixel_sha256, dhash=f"{record.dhash:016x}")
            rows.append(row)
        except Exception as exc:
            failures.append({**row, "quality_exclusion": type(exc).__name__, "detail": str(exc)[:200]})
        if index % 200 == 0 or index == len(specs):
            print(f"PREPARING={index}/{len(specs)} READABLE={len(rows)}", flush=True)

    groups = defaultdict(list)
    for record in records:
        groups[record.pixel_sha256].append(record)
    exact_groups = [[r.key for r in group] for group in groups.values() if len(group) > 1]
    pairs = []
    for a, b in combinations(records, 2):
        exact = a.pixel_sha256 == b.pixel_sha256
        distance = (a.dhash ^ b.dhash).bit_count()
        if not exact and (distance > 4 or abs((a.size[0] / a.size[1]) / (b.size[0] / b.size[1]) - 1) > .02):
            continue
        pairs.append({"ids": [a.key, b.key], "kind": "exact_image" if exact else "near_candidate",
                      "dhash_distance": distance, "cross_split": a.split != b.split})
    pairs.sort(key=lambda p: (not p["cross_split"], p["kind"] != "exact_image", p["dhash_distance"], p["ids"]))
    for index, pair in enumerate(pairs, 1):
        pair["id"] = f"pair:{index:04d}"
    manifest = {
        "schema_version": 1, "created_utc": utc_now(), "source_commit": COMMIT,
        "source_repo": str(repo), "rows": rows, "quality_exclusions": failures,
        "exact_image_groups": exact_groups, "pairs": pairs,
        "notes": {
            "originals_modified": False, "raw_data_uploaded": False,
            "political_screening": "PENDING_HUMAN_IMAGE_AND_TEXT_REVIEW",
            "labels_hidden_from_review_ui": True,
            "source_label_is_private_metadata_not_model_input": True,
            "split_changes": "NONE; no training dataset created",
            "dhash_is_only_a_candidate_signal": True,
            "duplicate_images_with_different_text_are_not_automatically_deduplicated": True,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    initialize_database(output / "review.sqlite3")
    print(f"REVIEW_PREPARED ROWS={len(rows)} QUALITY_EXCLUSIONS={len(failures)} PAIRS={len(pairs)}", flush=True)
    print(f"SAVED={output}\nNO_TRAINING_SPLITS_CREATED; SOURCE_FILES_UNCHANGED", flush=True)


class ReviewStore:
    def __init__(self, directory):
        self.directory = directory.resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1 or self.manifest.get("source_commit") != COMMIT:
            raise ValueError("Unsupported review manifest")
        self.repo = Path(self.manifest["source_repo"]).resolve()
        self.rows = {row["id"]: row for row in self.manifest["rows"]}
        self.pairs = {pair["id"]: pair for pair in self.manifest["pairs"]}
        if len(self.rows) != len(self.manifest["rows"]) or len(self.pairs) != len(self.manifest["pairs"]):
            raise ValueError("Duplicate review IDs")
        if any(key not in self.rows for pair in self.pairs.values() for key in pair["ids"]):
            raise ValueError("Pair references an unknown image")
        self.database = self.directory / "review.sqlite3"
        if not self.database.is_file():
            raise ValueError("Review database missing; do not recreate or overwrite it")
        self.manifest_sha256 = hashlib.sha256((self.directory / "manifest.json").read_bytes()).hexdigest()

    def decisions(self):
        with closing(sqlite3.connect(self.database, timeout=10)) as db, db:
            records = db.execute("SELECT kind,target,decision,note,updated_utc FROM decisions").fetchall()
        return [dict(zip(("kind", "target", "decision", "note", "updated_utc"), record)) for record in records]

    def save(self, kind, target, decision, note=""):
        if kind not in {"content", "pair"}:
            raise ValueError("Invalid review kind")
        targets = self.rows if kind == "content" else self.pairs
        choices = CONTENT_CHOICES if kind == "content" else PAIR_CHOICES
        if target not in targets or decision not in choices or not isinstance(note, str) or len(note) > 1000:
            raise ValueError("Invalid target, decision, or note")
        record = (kind, target, decision, note, utc_now())
        with closing(sqlite3.connect(self.database, timeout=10)) as db, db:
            db.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?)", record)
            db.execute("INSERT INTO events(kind,target,decision,note,updated_utc) VALUES (?,?,?,?,?)", record)

    def status(self):
        decisions = self.decisions()
        counts = {kind: Counter(d["decision"] for d in decisions if d["kind"] == kind)
                  for kind in ("content", "pair")}
        return {
            "readable_rows": len(self.rows), "quality_exclusions": len(self.manifest["quality_exclusions"]),
            "content_reviewed": sum(counts["content"].values()),
            "content_pending": len(self.rows) - sum(counts["content"].values()),
            "content_decisions": dict(counts["content"]), "pairs_total": len(self.pairs),
            "pairs_pending": len(self.pairs) - sum(counts["pair"].values()),
            "pair_decisions": dict(counts["pair"]), "training_ready": False,
            "reason": "Review metadata only; eligibility/group-aware split must be finalized separately.",
        }

    def public_state(self):
        # Gold harm labels, image hashes and source directory metadata stay out of the UI.
        return {
            "rows": [{key: row[key] for key in ("id", "text", "size")} for row in self.rows.values()],
            "pairs": list(self.pairs.values()), "decisions": self.decisions(), "status": self.status(),
        }

    def preview(self, key):
        if key not in self.rows:
            raise ValueError("Unknown readable image")
        row = self.rows[key]
        path = safe_image(self.repo, row["image_relative"])
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["file_sha256"]:
                raise ValueError("Source image changed after preparation")
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Multi-frame image is not eligible for this viewer")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=94)
        return buffer.getvalue()

    def export(self, target):
        target = target.resolve()
        if target.exists() or target.is_relative_to(self.repo):
            raise ValueError("Export exists or would write into source repository")
        snapshot = {"manifest_sha256": self.manifest_sha256, "exported_utc": utc_now(),
                    "status": self.status(), "decisions": self.decisions(),
                    "warning": "Human eligibility/duplicate decisions only; NOT training inputs or new harm labels."}
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"REVIEW_SNAPSHOT_SAVED={target}")


PAGE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LT-EDI 数据资格审核</title>
<style>body{font:16px system-ui;margin:20px auto;padding:0 20px;max-width:1200px;background:#f5f7fa;color:#17202a}
button,select,textarea,input{font:inherit;padding:9px;margin:5px}button{cursor:pointer}button:disabled{cursor:wait}
.cards{display:flex;gap:18px}.card{flex:1;min-width:0;background:white;padding:15px;border:1px solid #cbd5e1}
img{display:block;max-width:100%;max-height:68vh;object-fit:contain;margin:auto;cursor:zoom-in}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}.warn{background:#fff1d6;padding:12px}
#message{min-height:24px;color:#14532d}textarea{width:90%;height:45px}small{color:#475569}
@media(max-width:700px){.cards{display:block}}</style>
<h2>图片＋文本：政治内容排除与重复核对</h2>
<p class="warn">原始图片和标签不会修改，数据不上传云端。保留仅表示本次未发现政治内容，<b>不表示内容无害</b>。
涉及政治人物、组织、事件或政治宣传等内容请排除；不能确定的请选“不确定，排除”。必须同时查看图片和文字。
仅做数据资格审核，不推断新风险标签。原始有害标签在页面中隐藏。</p>
<div id="status"></div>
<select id="mode"><option value="content">逐条政治内容筛查</option><option value="pair">重复/近重复核对</option></select>
<select id="filter"><option value="pending">仅未审核</option><option value="all">全部（可修改已有决定）</option></select>
<button id="prev">上一条</button><button id="next">下一条（不保存）</button>
<input id="jump" type="number" min="1" style="width:80px"><button id="go">跳转</button>
<div id="position"></div><p id="pairinfo"></p><div id="cards" class="cards"></div>
<p><small>点击图片在新标签页放大。若与文字明显不一致或无法辨认，可备注并选择不确定。</small></p>
<textarea id="note" maxlength="1000" placeholder="可选：排除/重复核对理由。不要在此创建风险标签。"></textarea>
<div id="actions"></div><div id="message"></div>
<script nonce="__NONCE__">
const token=__TOKEN__;let state,queue=[],index=0,ready=false,busy=false,generation=0;
const $=id=>document.getElementById(id),decisions=new Map();
const choices={content:[['keep','1 保留：未发现政治内容'],['exclude_political','2 排除：涉及政治'],['exclude_uncertain','3 不确定：排除']],
pair:[['same_content','1 相同图文内容'],['same_template','2 同模板、文字/语义不同'],['different_content','3 不同内容'],['uncertain','4 不确定，待处理']]};
function status(){let c=0,p=0,keep=0;for(const d of decisions.values()){if(d.kind==='content'){c++;keep+=d.decision==='keep';}else p++;}
$('status').textContent=`图文审核 ${c}/${state.rows.length}；其中保留 ${keep}。重复核对 ${p}/${state.pairs.length}。质量排除 ${state.status.quality_exclusions}。仅保存审核结果，尚未生成训练划分。`;}
function rebuild(preferred){const mode=$('mode').value;queue=(mode==='content'?state.rows:state.pairs)
.filter(x=>$('filter').value==='all'||!decisions.has(mode+'|'+x.id));
index=preferred?Math.max(0,queue.findIndex(x=>x.id===preferred)):0;render();}
function disable(){for(const b of $('actions').querySelectorAll('button'))b.disabled=!ready||busy;
for(const id of ['mode','filter','prev','next','go','jump'])$(id).disabled=busy;}
async function render(){const current=++generation;ready=false;$('cards').replaceChildren();$('actions').replaceChildren();$('message').textContent='';$('note').value='';
const mode=$('mode').value,item=queue[index];$('pairinfo').textContent='';$('jump').value=queue.length?index+1:'';
$('position').textContent=queue.length?`当前 ${index+1}/${queue.length}，ID ${item.id}`:'当前队列已完成（可切换“全部”查看已保存结果）。';status();if(!item)return;
const saved=decisions.get(mode+'|'+item.id);if(saved){$('note').value=saved.note;$('message').textContent='已保存：'+saved.decision;}
if(mode==='pair')$('pairinfo').textContent=`${item.kind==='exact_image'?'完全相同像素图片（仍须比较转录文字）':'dHash 近重复候选，不代表真正重复'}；${item.cross_split?'跨原始 train/dev':'同原始集合'}；距离 ${item.dhash_distance}。同模板不同文字不能直接当成同一条删除。`;
const rows=mode==='content'?[item]:item.ids.map(id=>state.rows.find(r=>r.id===id));let loaded=0;
for(const row of rows){const card=document.createElement('section');card.className='card';const title=document.createElement('p');title.textContent=row.id+' · '+row.size.join('×');
const img=document.createElement('img');img.alt='待审核图片 '+row.id;const url='/image?id='+encodeURIComponent(row.id);
img.onload=()=>{if(generation===current&&++loaded===rows.length){ready=true;disable();}};
img.onerror=()=>{if(generation===current)$('message').textContent='图片加载失败，请勿审核；查看服务器日志。';};
img.onclick=()=>window.open(url,'_blank','noopener');img.src=url;
const text=document.createElement('pre');text.textContent='转录文字：\n'+row.text;card.append(title,img,text);$('cards').append(card);}
for(const [choice,label] of choices[mode]){const b=document.createElement('button');b.textContent=label;b.disabled=true;b.onclick=()=>save(choice);$('actions').append(b);}}
async function save(choice){if(busy||!ready||!queue[index])return;busy=true;disable();const mode=$('mode').value,item=queue[index];
const data={kind:mode,target:item.id,decision:choice,note:$('note').value};
try{const response=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify(data)});
const result=await response.json();if(!response.ok)throw Error(result.error||'保存失败');decisions.set(mode+'|'+item.id,data);
if($('filter').value==='pending'){queue.splice(index,1);index=Math.min(index,Math.max(0,queue.length-1));}else index=Math.min(index+1,queue.length-1);
render();window.scrollTo(0,0);}catch(error){$('message').textContent='未保存：'+error.message;}finally{busy=false;disable();}}
for(const id of ['mode','filter'])$(id).onchange=()=>{if(!busy)rebuild();};
$('prev').onclick=()=>{if(!busy&&index>0){index--;render();}};$('next').onclick=()=>{if(!busy&&index+1<queue.length){index++;render();}};
$('go').onclick=()=>{const n=Number($('jump').value);if(!busy&&Number.isInteger(n)&&n>=1&&n<=queue.length){index=n-1;render();}};
document.addEventListener('keydown',e=>{if(['TEXTAREA','INPUT','SELECT'].includes(e.target.tagName)||e.ctrlKey||e.altKey||e.metaKey||e.repeat)return;
const n=Number(e.key)-1;if(n>=0&&n<choices[$('mode').value].length){e.preventDefault();save(choices[$('mode').value][n][0]);}});
fetch('/api/state').then(r=>{if(!r.ok)throw Error('读取失败');return r.json();}).then(data=>{state=data;for(const d of data.decisions)decisions.set(d.kind+'|'+d.target,d);rebuild();})
.catch(e=>{$('message').textContent=e.message;});
</script></html>"""


def handler_for(store):
    token, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(20)
    page = PAGE.replace("__NONCE__", nonce).replace("__TOKEN__", json.dumps(token)).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No image/text/request-body logs.

        def send_payload(self, status, payload, content_type="application/json; charset=utf-8"):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", f"default-src 'self'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; img-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def allowed_host(self):
            try:
                return urlsplit("http://" + self.headers.get("Host", "")).hostname in {"localhost", "127.0.0.1", "::1"}
            except ValueError:
                return False

        def do_GET(self):
            if not self.allowed_host():
                return self.send_payload(403, {"error": "Loopback Host required"})
            url = urlsplit(self.path)
            try:
                if url.path == "/":
                    return self.send_payload(200, page, "text/html; charset=utf-8")
                if url.path == "/api/state":
                    return self.send_payload(200, store.public_state())
                if url.path == "/image":
                    key = parse_qs(url.query).get("id", [""])[0]
                    return self.send_payload(200, store.preview(key), "image/jpeg")
                return self.send_payload(404, {"error": "Not found"})
            except Exception as exc:
                print(f"VIEWER_READ_ERROR={type(exc).__name__}: {exc}", flush=True)
                return self.send_payload(400, {"error": str(exc)[:200]})

        def do_POST(self):
            if not self.allowed_host() or self.headers.get("X-Review-Token") != token:
                return self.send_payload(403, {"error": "Private review token required"})
            if urlsplit(self.path).path != "/api/decision":
                return self.send_payload(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request length")
                data = json.loads(self.rfile.read(length))
                store.save(data["kind"], data["target"], data["decision"], data.get("note", ""))
                return self.send_payload(200, {"saved": True})
            except Exception as exc:
                return self.send_payload(400, {"error": str(exc)[:200]})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Create new review metadata; never modifies raw files")
    prep.add_argument("--repo", type=Path, default=Path(DEFAULT_REPO))
    prep.add_argument("--output", type=Path, default=Path(DEFAULT_REVIEW))
    for command in ("serve", "status", "export"):
        p = sub.add_parser(command)
        p.add_argument("--review-dir", type=Path, default=Path(DEFAULT_REVIEW))
        if command == "serve":
            p.add_argument("--port", type=int, default=8765)
        if command == "export":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.repo, args.output)
        return 0
    store = ReviewStore(args.review_dir)
    if args.command == "status":
        print(json.dumps(store.status(), ensure_ascii=False, indent=2))
    elif args.command == "export":
        store.export(args.output)
    else:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(store))
        print(f"REVIEW_SERVER=http://127.0.0.1:{server.server_port}", flush=True)
        print("PRIVATE_SSH_PORT_FORWARDING_ONLY; NO_GPU; NO_CLOUD_UPLOAD; decisions auto-saved in review.sqlite3", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("REVIEW_SERVER_STOPPED; saved decisions retained", flush=True)
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"REVIEW_STOPPED {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(2)
