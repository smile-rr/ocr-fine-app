"""生成 4 个 step-by-step notebook (.ipynb)。

运行一次即可：python scripts/make_notebooks.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def save(name: str, cells: list[dict]):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NB_DIR / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path}")


# ------------------------------------------------------------------
# 01 数据探索
# ------------------------------------------------------------------
nb01 = [
    md("# 01 · 数据探索\n\n**目标**：看清楚每个数据集长啥样，筛出可用样本。\n\n"
       "本 notebook 用到：\n"
       "- `bsmock/pubtables-1m` — 学术论文表格\n"
       "- `bsmock/FinTabNet.c` — 金融表格\n"
       "- `ByteDance/ComTQA` — 表格问答\n"
       "- `data/samples/*.pdf` — 手工测试 PDF\n"),
    code("import sys, pathlib\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
         "from src import config as C\n"
         "from datasets import load_from_disk\n"
         "import pandas as pd\n"),
    md("## 1. 载入 PubTables-1M 子集"),
    code("ds = load_from_disk(str(C.RAW_DIR / 'pubtables_sample'))\n"
         "print(len(ds), ds.features)\n"
         "ds[0]"),
    md("## 2. 看一张表格图 + 对应 HTML"),
    code("sample = ds[0]\n"
         "display(sample['image'])\n"
         "print(sample.get('html', '')[:500])"),
    md("## 3. 载入 ComTQA QA 对"),
    code("qa_ds = load_from_disk(str(C.RAW_DIR / 'comtqa_sample'))\n"
         "print(len(qa_ds), qa_ds.features)\n"
         "qa_ds[0]"),
    md("## 4. 测试手工 PDF 的表格抽取\n\n用 pdfplumber 先拿到 ground truth，后面 Stage1 会训练 VLM 学会做这件事。"),
    code("from src.pdf_utils import pdf_to_images, extract_tables\n"
         "pdfs = list(C.SAMPLES_DIR.glob('*.pdf'))\n"
         "pdfs"),
    code("pdf = pdfs[0]\n"
         "imgs = pdf_to_images(pdf, C.DATA_DIR / 'preview' / pdf.stem)\n"
         "print(f'{len(imgs)} pages')\n"
         "tables = extract_tables(pdf)\n"
         "print(f'{len(tables)} tables')\n"
         "print(tables[0]['markdown'][:400] if tables else '无表格')"),
    md("## 5. 构造训练集\n\n运行脚本：\n\n```bash\npython scripts/prepare_stage1.py\npython scripts/prepare_stage2.py\n```\n\n输出：\n- `data/stage1_train/{train,val}.jsonl`\n- `data/stage2_train/{train,val}.jsonl`\n"),
    code("from src.data import load_jsonl\n"
         "s1 = load_jsonl(C.DATA_DIR / 'stage1_train' / 'train.jsonl')\n"
         "s2 = load_jsonl(C.DATA_DIR / 'stage2_train' / 'train.jsonl')\n"
         "print(f'stage1={len(s1)}, stage2={len(s2)}')\n"
         "print('样例 stage1:', s1[0])\n"
         "print('样例 stage2:', s2[0])"),
]
save("01_explore_data.ipynb", nb01)


# ------------------------------------------------------------------
# 02 Stage1 VLM 微调
# ------------------------------------------------------------------
nb02 = [
    md("# 02 · Stage 1 · VLM 微调（图→Markdown 表格）\n\n"
       "**模型**：`mlx-community/Qwen2-VL-2B-Instruct-4bit`（4-bit 量化，~3GB）\n\n"
       "**框架**：MLX-VLM（Apple Silicon 原生，8GB 也能跑）\n\n"
       "**备选**：Colab 用 Unsloth（见文末）\n"),
    md("## 0. 检查环境"),
    code("import platform, torch\n"
         "print('Python:', platform.python_version())\n"
         "try:\n"
         "    import mlx.core as mx\n"
         "    print('MLX:', mx.__version__, '| device:', mx.default_device())\n"
         "except Exception as e:\n"
         "    print('MLX 未安装:', e)\n"),
    md("## 1. 转换数据格式为 MLX-VLM 期望格式\n\n"
       "MLX-VLM 的 `lora.py` 期望 jsonl，每行 `{\"messages\": [...], \"images\": [path]}`。\n"
       "我们已经在 `prepare_stage1.py` 里产出 sharegpt 格式，稍作转换。"),
    code("import json, sys, pathlib\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
         "from src import config as C\n"
         "from src.data import load_jsonl\n"
         "\n"
         "def to_mlx_vlm(rows):\n"
         "    out = []\n"
         "    for r in rows:\n"
         "        msgs = r['messages']\n"
         "        img = next(c['image'] for c in msgs[0]['content'] if c['type']=='image')\n"
         "        text = next(c['text'] for c in msgs[0]['content'] if c['type']=='text')\n"
         "        out.append({\n"
         "            'images': [str(C.ROOT / img)],\n"
         "            'messages': [\n"
         "                {'role': 'user', 'content': text},\n"
         "                {'role': 'assistant', 'content': msgs[1]['content']},\n"
         "            ]\n"
         "        })\n"
         "    return out\n"
         "\n"
         "train = to_mlx_vlm(load_jsonl(C.DATA_DIR/'stage1_train'/'train.jsonl'))\n"
         "val   = to_mlx_vlm(load_jsonl(C.DATA_DIR/'stage1_train'/'val.jsonl'))\n"
         "out_dir = C.DATA_DIR / 'stage1_mlx'\n"
         "out_dir.mkdir(parents=True, exist_ok=True)\n"
         "for name, rows in [('train', train), ('valid', val)]:\n"
         "    with (out_dir / f'{name}.jsonl').open('w', encoding='utf-8') as f:\n"
         "        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\\n')\n"
         "print('done:', out_dir)"),
    md("## 2. 启动 LoRA 训练（MLX-VLM CLI）\n\n"
       "在终端执行（或用下方 `!` 运行）：\n\n"
       "```bash\npython -m mlx_vlm.lora \\\n"
       "    --model mlx-community/Qwen2-VL-2B-Instruct-4bit \\\n"
       "    --train \\\n"
       "    --data data/stage1_mlx \\\n"
       "    --iters 300 \\\n"
       "    --batch-size 1 \\\n"
       "    --lora-layers 8 \\\n"
       "    --learning-rate 1e-4 \\\n"
       "    --adapter-path models/stage1_adapter\n```\n\n"
       "**预期显存**：~6–8GB。300 iter 在 M2 Pro 约 20–30 分钟。"),
    code("# 可选：在 notebook 内直接跑（会阻塞 kernel）\n"
         "# !python -m mlx_vlm.lora --model mlx-community/Qwen2-VL-2B-Instruct-4bit --train \\\n"
         "#     --data ../data/stage1_mlx --iters 100 --batch-size 1 --lora-layers 8 \\\n"
         "#     --learning-rate 1e-4 --adapter-path ../models/stage1_adapter"),
    md("## 3. 推理对比（微调前 vs 微调后）"),
    code("from src.infer import extract_table_from_image\n"
         "import os\n"
         "\n"
         "img = sorted((C.DATA_DIR/'stage1_images').glob('*.png'))[0]\n"
         "print('=== 微调前 ===')\n"
         "print(extract_table_from_image(img, adapter=None, max_tokens=256))\n"
         "print('\\n=== 微调后 ===')\n"
         "print(extract_table_from_image(img, adapter=str(C.STAGE1_ADAPTER), max_tokens=256))"),
    md("## 4. Colab 备选方案\n\n"
       "若本地太慢，打开 `notebooks/02b_stage1_colab.ipynb`（待补，用 Unsloth + bitsandbytes 4bit）。"),
]
save("02_finetune_stage1.ipynb", nb02)


# ------------------------------------------------------------------
# 03 Stage2 LLM 微调
# ------------------------------------------------------------------
nb03 = [
    md("# 03 · Stage 2 · LLM 微调（表格 QA）\n\n"
       "**模型**：`mlx-community/Qwen2.5-0.5B-Instruct-4bit` (~400MB)\n\n"
       "**速度**：M2 Pro 约 10–15 分钟 2000 样本 2 epoch。"),
    code("import sys, pathlib, json\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
         "from src import config as C\n"
         "from src.data import load_jsonl\n"),
    md("## 1. 转成 MLX-LM 格式\n\n"
       "`mlx_lm.lora` 对 chat 模型期望：`{\"messages\": [{\"role\":..., \"content\":...}]}`"),
    code("def alpaca_to_chat(r):\n"
         "    user = f\"{r['instruction']}\\n\\n{r['input']}\"\n"
         "    return {'messages': [\n"
         "        {'role': 'user', 'content': user},\n"
         "        {'role': 'assistant', 'content': r['output']},\n"
         "    ]}\n"
         "\n"
         "for name in ['train', 'val']:\n"
         "    rows = [alpaca_to_chat(r) for r in load_jsonl(C.DATA_DIR/'stage2_train'/f'{name}.jsonl')]\n"
         "    out = C.DATA_DIR / 'stage2_mlx'\n"
         "    out.mkdir(parents=True, exist_ok=True)\n"
         "    with (out / (f'{\"valid\" if name==\"val\" else name}.jsonl')).open('w', encoding='utf-8') as f:\n"
         "        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\\n')\n"
         "print('done')"),
    md("## 2. 启动 LoRA 训练\n\n```bash\n"
       "python -m mlx_lm.lora \\\n"
       "    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \\\n"
       "    --train \\\n"
       "    --data data/stage2_mlx \\\n"
       "    --iters 600 \\\n"
       "    --batch-size 2 \\\n"
       "    --num-layers 8 \\\n"
       "    --learning-rate 2e-4 \\\n"
       "    --adapter-path models/stage2_adapter\n```"),
    md("## 3. 推理对比"),
    code("from src.infer import chat\n"
         "\n"
         "table_md = \"| 年份 | 营收(亿) | 净利润(亿) |\\n|---|---|---|\\n| 2022 | 100 | 15 |\\n| 2023 | 120 | 18 |\\n| 2024 | 135 | 22 |\"\n"
         "q = '哪一年净利润同比增长最大？'\n"
         "msgs = [{'role': 'system', 'content': '你是表格问答助手'},\n"
         "        {'role': 'user', 'content': f'表格:\\n{table_md}\\n问题:{q}'}]\n"
         "print('=== base ===')\n"
         "print(chat(msgs, adapter=None))\n"
         "print('\\n=== finetuned ===')\n"
         "print(chat(msgs, adapter=str(C.STAGE2_ADAPTER)))"),
    md("## 4. 批量评估"),
    code("from src.eval import exact_match, token_f1, rouge_l\n"
         "from statistics import mean\n"
         "\n"
         "val = load_jsonl(C.DATA_DIR/'stage2_train'/'val.jsonl')[:30]\n"
         "ems, f1s, rs = [], [], []\n"
         "for r in val:\n"
         "    msgs = [{'role':'user', 'content': f\"{r['instruction']}\\n\\n{r['input']}\"}]\n"
         "    pred = chat(msgs, adapter=str(C.STAGE2_ADAPTER), max_tokens=256)\n"
         "    ems.append(exact_match(pred, r['output']))\n"
         "    f1s.append(token_f1(pred, r['output']))\n"
         "    rs.append(rouge_l(pred, r['output']))\n"
         "print(f'EM={mean(ems):.3f}  F1={mean(f1s):.3f}  ROUGE-L={mean(rs):.3f}')"),
]
save("03_finetune_stage2.ipynb", nb03)


# ------------------------------------------------------------------
# 04 端到端 RAG demo
# ------------------------------------------------------------------
nb04 = [
    md("# 04 · 端到端 RAG Demo\n\n上传 PDF → VLM 抽表 → 清洗 → 向量库 → 检索 → LLM 回答。"),
    code("import sys, pathlib\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
         "from src import config as C\n"
         "from src.pdf_utils import pdf_to_images, extract_tables\n"
         "from src.infer import extract_table_from_image, chat\n"
         "from src.rag import TableVectorStore, parse_markdown_table, build_rag_prompt\n"),
    md("## 1. 选一个测试 PDF"),
    code("pdf = sorted(C.SAMPLES_DIR.glob('*.pdf'))[0]\n"
         "doc_id = pdf.stem\n"
         "print(pdf)"),
    md("## 2. 两种抽表方式并存：pdfplumber (fast/structured) + VLM (robust/image)"),
    code("# 方式 A：pdfplumber (快，有结构时最准)\n"
         "pp_tables = extract_tables(pdf)\n"
         "print(f'pdfplumber 抽到 {len(pp_tables)} 张表')\n"
         "if pp_tables:\n"
         "    print(pp_tables[0]['markdown'][:300])"),
    code("# 方式 B：VLM 从图片抽（扫描件或结构异常时更鲁棒）\n"
         "imgs = pdf_to_images(pdf, C.DATA_DIR / 'preview' / doc_id)\n"
         "vlm_md = extract_table_from_image(imgs[0], adapter=str(C.STAGE1_ADAPTER), max_tokens=512)\n"
         "print(vlm_md[:500])"),
    md("## 3. 写入向量库"),
    code("store = TableVectorStore()\n"
         "for t in pp_tables:\n"
         "    df = parse_markdown_table(t['markdown'])\n"
         "    if not df.empty:\n"
         "        n = store.add(df, doc_id=doc_id, page=t['page'])\n"
         "        print(f'p{t[\"page\"]}  +{n} chunks')\n"
         "print('总 chunks:', store.count())"),
    md("## 4. RAG 问答"),
    code("question = '营收最高的是哪一年，具体是多少？'\n"
         "hits = store.search(question, top_k=5, doc_filter=doc_id)\n"
         "for h in hits: print(f'{h[\"score\"]:.2f}  {h[\"text\"][:120]}')\n"
         "\n"
         "msgs = build_rag_prompt(question, hits)\n"
         "answer = chat(msgs, adapter=str(C.STAGE2_ADAPTER), max_tokens=400)\n"
         "print('\\n=== 回答 ===')\n"
         "print(answer)"),
    md("## 5. 打开 Streamlit UI\n\n```bash\nstreamlit run app/streamlit_app.py\n```"),
]
save("04_end_to_end_rag.ipynb", nb04)

print("\nAll notebooks generated. Open with: jupyter lab notebooks/")
