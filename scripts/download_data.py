"""一键下载：HuggingFace 数据集 + 手工测试 PDF 样本。

默认走 **streaming 模式**：下载量严格按条数走，不会拉整个 shard。
对 PubTables-1M 这种 WebDataset 分片大数据集尤其有效（非 streaming 模式
即便切片 500 条也会拉 2–6 GB 的 shard；streaming 只取你要的字节）。

用法：
    # 默认：streaming 拉小量样本
    python scripts/download_data.py

    # 关闭 streaming，用原始 load_dataset（整 shard 下载，测试/调试用）
    python scripts/download_data.py --no-stream

    # 只下 HF / 只下 PDF
    python scripts/download_data.py --skip-pdf
    python scripts/download_data.py --skip-hf

    # 自定义采样量（0 表示跳过该集）
    python scripts/download_data.py --pubtabnet 100 --fintabnet 0 --comtqa 500
    # 注：--pubtables 保留为 --pubtabnet 的 alias（老命令仍可用）

    # 清理历史下载（任何 --clean* 都是「只清不下」，完事就退出）
    python scripts/download_data.py --clean            # 删 data/raw/*_sample/
    python scripts/download_data.py --clean-cache      # 删 HF shard cache
    python scripts/download_data.py --clean-all        # 样本 + cache + PDF 全删
    python scripts/download_data.py --clean --dry-run  # 只预览要删什么

    # 想清理后重下 → 分两步（或用 --fresh 一步到位）
    python scripts/download_data.py --fresh                      # 等价于 clean-all + 下载
    python scripts/download_data.py --fresh --pubtables 100      # fresh 也接采样参数

环境变量（国内加速）：
    HF_ENDPOINT=https://hf-mirror.com
    HF_HUB_ENABLE_HF_TRANSFER=1       # 需 `pip install hf_transfer`
"""
from __future__ import annotations
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("download_data")


# 手工测试 PDF（公开金融/科技报告，用于烟囱测试 + UI demo）
SAMPLE_PDFS = [
    ("apple_2023_q4.pdf",
     "https://www.apple.com/newsroom/pdfs/fy2023-q4/FY23_Q4_Consolidated_Financial_Statements.pdf"),
    ("nvidia_2024_q2.pdf",
     "https://s201.q4cdn.com/141608511/files/doc_financials/2024/q2/NVDA-F2Q24-CFO-Commentary.pdf"),
    ("irs_form_1040.pdf",
     "https://www.irs.gov/pub/irs-pdf/f1040.pdf"),
]


def _dir_size(p: Path) -> int:
    """递归算目录总字节数；不存在返回 0。"""
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _clean(args) -> None:
    """根据 --clean / --clean-cache / --clean-all 删历史产物。

    --clean       → data/raw/{pubtabnet,fintabnet,comtqa}_sample/
    --clean-cache → data/hf_cache/（整 shard 下载留下的大缓存）
    --clean-all   → 以上全部 + data/samples/*.pdf + 旧 pubtables_sample/
    --dry-run     → 只打印要删什么、多少 MB，不真删
    """
    do_samples = args.clean or args.clean_all
    do_cache = args.clean_cache or args.clean_all
    do_pdfs = args.clean_all

    targets: list[Path] = []
    if do_samples:
        # 只删本次 N>0 的那几个（避免误删用户自建的其它子目录）
        if args.pubtabnet > 0 or args.clean_all:
            targets.append(C.RAW_DIR / "pubtabnet_sample")
        if args.fintabnet > 0 or args.clean_all:
            targets.append(C.RAW_DIR / "fintabnet_sample")
        if args.comtqa > 0 or args.clean_all:
            targets.append(C.RAW_DIR / "comtqa_sample")
        if args.clean_all:
            # 迁移遗留：旧 PubTables-1M 的失败产物
            targets.append(C.RAW_DIR / "pubtables_sample")
    if do_cache:
        targets.append(C.HF_CACHE)
    if do_pdfs:
        # PDF 是通过 urlretrieve 下载的，可能有 .part 临时文件
        for name, _ in SAMPLE_PDFS:
            targets.append(C.SAMPLES_DIR / name)
        # 也清一下 HF cache 里的 .incomplete
        targets += list(C.HF_CACHE.rglob("*.incomplete")) if C.HF_CACHE.exists() else []

    if not targets:
        return

    total = 0
    for t in targets:
        sz = _dir_size(t)
        total += sz
        if not t.exists():
            log.info(f"  · {t}  (不存在，跳过)")
            continue
        action = "would remove" if args.dry_run else "removing"
        log.info(f"  · {action} {t}  ({_fmt_size(sz)})")
        if args.dry_run:
            continue
        try:
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink()
        except Exception as e:
            log.error(f"    !! 删除失败：{e}")

    tag = "would free" if args.dry_run else "freed"
    log.info(f"🧹 clean 完成：{tag} {_fmt_size(total)}")


def _log_env_hints():
    ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    xfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
    log.info(f"HF_ENDPOINT={ep}  HF_HUB_ENABLE_HF_TRANSFER={xfer}")
    if "hf-mirror" not in ep:
        log.info("  提示：国内可 `export HF_ENDPOINT=https://hf-mirror.com` 加速")


def _stream_take(repo: str, split: str, n: int, save_to: Path) -> int:
    """Streaming 拉 n 条，转成常规 Dataset 存到磁盘。

    相比 `load_dataset(split='train[:n]')`：只下载实际需要的字节，
    不会触发整个 shard 的下载。
    """
    from datasets import Dataset, load_dataset
    from tqdm import tqdm

    stream = load_dataset(repo, split=split, streaming=True)
    features = getattr(stream, "features", None)  # 可能为 None

    rows: list[dict] = []
    for ex in tqdm(stream.take(n), total=n, desc=repo, unit="ex"):
        rows.append(ex)
    if not rows:
        log.warning(f"  !! {repo} 没拉到任何样本")
        return 0

    ds = Dataset.from_list(rows, features=features) if features else Dataset.from_list(rows)
    save_to.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(save_to))
    return len(ds)


def _bulk_download(repo: str, split: str, n: int, save_to: Path) -> int:
    """原始 load_dataset 路径：切片下载（会拉整 shard）。"""
    from datasets import load_dataset
    ds = load_dataset(repo, split=f"{split}[:{n}]", cache_dir=str(C.HF_CACHE))
    save_to.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(save_to))
    return len(ds)


def download_hf_datasets(args):
    # 这三个都是「image + html/structure」配对好的 parquet 数据集，
    # 不再是多 shard WebDataset（老的 bsmock/pubtables-1m 默认会拉到
    # 只含 XML 的 annotation shard，对 VLM 训练没用）。
    jobs = [
        ("pubtabnet",        "apoidea/pubtabnet-html",  "train", args.pubtabnet,
         C.RAW_DIR / "pubtabnet_sample"),
        ("fintabnet-otsl",   "ds4sd/FinTabNet_OTSL",    "train", args.fintabnet,
         C.RAW_DIR / "fintabnet_sample"),
        ("comtqa",           "ByteDance/ComTQA",        "train", args.comtqa,
         C.RAW_DIR / "comtqa_sample"),
    ]
    mode = "bulk (load_dataset)" if args.no_stream else "streaming"
    log.info(f"HF datasets mode: {mode}")

    for name, repo, split, n, save_to in jobs:
        if n <= 0:
            log.info(f"skip {name} (n=0)")
            continue
        if save_to.exists() and any(save_to.iterdir()):
            log.info(f"exists: {save_to} (删掉可重下)")
            continue
        try:
            log.info(f"↓ {name} · {n} samples · {repo}")
            if args.no_stream:
                saved = _bulk_download(repo, split, n, save_to)
            else:
                saved = _stream_take(repo, split, n, save_to)
            log.info(f"  ✓ saved {saved} rows → {save_to}")
        except Exception as e:
            log.error(f"  !! failed {name}: {e}")
            log.info("  提示：HF_ENDPOINT=https://hf-mirror.com 或 --no-stream 重试")


def download_sample_pdfs():
    C.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLE_PDFS:
        target = C.SAMPLES_DIR / name
        if target.exists():
            log.info(f"exists: {target.name}")
            continue
        try:
            log.info(f"↓ {name}")
            urlretrieve(url, target)
        except Exception as e:
            log.error(f"  !! {name} failed: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pubtabnet", "--pubtables", dest="pubtabnet",
                   type=int, default=500,
                   help="PubTabNet (apoidea/pubtabnet-html) 样本数（0 跳过）")
    p.add_argument("--fintabnet", type=int, default=300,
                   help="FinTabNet_OTSL (ds4sd/FinTabNet_OTSL) 样本数（0 跳过）")
    p.add_argument("--comtqa",    type=int, default=1000,
                   help="ComTQA 样本数（0 跳过）")
    p.add_argument("--skip-hf",   action="store_true",
                   help="跳过 HuggingFace 数据集")
    p.add_argument("--skip-pdf",  action="store_true",
                   help="跳过样例 PDF")
    p.add_argument("--no-stream", action="store_true",
                   help="禁用 streaming，用整 shard 下载（慢，但能 resume）")
    p.add_argument("--clean", action="store_true",
                   help="只清 data/raw/*_sample/（本次 N>0 的集），清完就退出")
    p.add_argument("--clean-cache", action="store_true",
                   help="只清 data/hf_cache/（整 shard 缓存，可能上 GB），清完就退出")
    p.add_argument("--clean-all", action="store_true",
                   help="只清 sample + HF cache + 样例 PDF，清完就退出")
    p.add_argument("--fresh", action="store_true",
                   help="先执行 clean-all，再重新下载（等价于 --clean-all 之后再跑一次）")
    p.add_argument("--dry-run", action="store_true",
                   help="与 --clean* / --fresh 搭配，只预览要删什么，不真删也不下载")
    return p.parse_args()


def main():
    args = parse_args()
    _log_env_hints()

    clean_only = args.clean or args.clean_cache or args.clean_all
    if clean_only or args.fresh:
        log.info("🧹 cleaning previous downloads...")
        # --fresh 走 clean-all 的逻辑
        if args.fresh:
            args.clean_all = True
        _clean(args)
        if args.dry_run:
            log.info("dry-run 结束（未下载）")
            return
        if clean_only and not args.fresh:
            log.info("✅ clean-only 模式，不下载。想重下请再跑一次（不加 --clean*）或用 --fresh")
            return

    if not args.skip_hf:
        download_hf_datasets(args)
    if not args.skip_pdf:
        download_sample_pdfs()
    log.info("DONE. 检查 data/raw/ 和 data/samples/")


if __name__ == "__main__":
    main()
