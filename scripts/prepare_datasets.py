#!/usr/bin/env python3
"""Prepare all datasets into the directory structure expected by the loaders.

Converts HuggingFace parquet datasets, extracts tarballs, and renders PDFs
into the canonical images/ + annotations/ layout.

Usage:
    python scripts/prepare_datasets.py          # prepare all
    python scripts/prepare_datasets.py D-02     # prepare one
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# D-02  FinTabNet  (tar.gz → images/ + annotations/)
# ---------------------------------------------------------------------------

def prepare_fintabnet():
    base = ROOT / "data" / "fintabnet"
    images_dir = base / "images"
    annotations_dir = base / "annotations"

    if images_dir.exists() and any(images_dir.iterdir()):
        print("  D-02 FinTabNet: images/ already exists, skipping")
        return

    # Step 1: extract tarballs if not already done
    structure_dir = base / "FinTabNet.c-Structure"
    annotations_src = base / "FinTabNet.c-PDF_Annotations"
    if not structure_dir.exists():
        structure_tar = base / "FinTabNet.c-Structure.tar.gz"
        annotations_tar = base / "FinTabNet.c-PDF_Annotations.tar.gz"
        if not structure_tar.exists():
            print("  D-02 FinTabNet: tar.gz not found, skipping")
            return
        print("  D-02 FinTabNet: extracting tarballs...")
        subprocess.run(["tar", "-xzf", str(structure_tar), "-C", str(base)], check=True)
        if annotations_tar.exists():
            subprocess.run(["tar", "-xzf", str(annotations_tar), "-C", str(base)], check=True)

    # Step 2: restructure flat images into {doc_id}/page_0.png
    # FinTabNet images are named like: A_2003_page_19_table_0.jpg
    raw_images_dir = structure_dir / "images"
    if not raw_images_dir.exists():
        print("  D-02 FinTabNet: extracted images/ not found, skipping")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print("  D-02 FinTabNet: restructuring images and annotations...")
    count = 0
    for img_path in sorted(raw_images_dir.glob("*.jpg")):
        doc_id = img_path.stem  # e.g. A_2003_page_19_table_0
        doc_dir = images_dir / doc_id
        doc_dir.mkdir(exist_ok=True)
        # Symlink as page_0.png (each image is a single table crop)
        dst = doc_dir / "page_0.png"
        if not dst.exists():
            img = Image.open(img_path)
            img.save(dst)

        # Convert annotation: {doc_id}_tables.json → {doc_id}.json
        # Annotation filename pattern: strip _table_N → _tables.json
        parts = doc_id.rsplit("_table_", 1)
        ann_src_name = f"{parts[0]}_tables.json"
        ann_src_path = annotations_src / ann_src_name
        ann_dst_path = annotations_dir / f"{doc_id}.json"
        if ann_src_path.exists() and not ann_dst_path.exists():
            with open(ann_src_path) as f:
                cells_data = json.load(f)
            # Build table HTML from cells
            table_idx = int(parts[1]) if len(parts) > 1 else 0
            cell_entry = cells_data[table_idx] if table_idx < len(cells_data) else cells_data[0] if cells_data else {}
            cells = cell_entry.get("cells", []) if isinstance(cell_entry, dict) else []
            html_parts = []
            for cell in cells:
                html_parts.append(cell.get("json_text_content", ""))
            ann = {"tables": [{"page": 0, "html": " | ".join(html_parts)}]}
            with open(ann_dst_path, "w") as f:
                json.dump(ann, f)

        count += 1
        if count % 10000 == 0:
            print(f"    {count} images processed...")

    print(f"  D-02 FinTabNet: ready ({count} documents)")


# ---------------------------------------------------------------------------
# D-03  FUNSD  (parquet → images/ + annotations/)
# ---------------------------------------------------------------------------

def prepare_funsd():
    base = ROOT / "data" / "funsd"
    images_dir = base / "images"
    annotations_dir = base / "annotations"

    if images_dir.exists() and any(images_dir.iterdir()):
        print("  D-03 FUNSD: images/ already exists, skipping")
        return

    parquet_dir = base / "funsd"
    parquets = sorted(parquet_dir.glob("*.parquet")) if parquet_dir.exists() else []
    if not parquets:
        print("  D-03 FUNSD: no parquet files found, skipping")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for pq in parquets:
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            raw_id = row["id"]
            doc_id = f"funsd_{int(raw_id):04d}" if str(raw_id).isdigit() else f"funsd_{raw_id}"
            # Save image
            img_data = row["image"]
            if isinstance(img_data, dict) and img_data.get("bytes"):
                img = Image.open(io.BytesIO(img_data["bytes"]))
                img.save(images_dir / f"{doc_id}.png")
            # Save annotations (FUNSD format: {"form": [{"text": ...}]})
            tokens = row["tokens"]
            ner_tags = row["ner_tags"]
            form_entries = []
            for token, tag in zip(tokens, ner_tags):
                form_entries.append({"text": str(token), "ner_tag": int(tag)})
            ann = {"form": form_entries}
            with open(annotations_dir / f"{doc_id}.json", "w") as f:
                json.dump(ann, f)
            count += 1

    print(f"  D-03 FUNSD: ready ({count} documents)")


# ---------------------------------------------------------------------------
# D-04  DocVQA  (parquet → images/ + annotations/)
# ---------------------------------------------------------------------------

def prepare_docvqa():
    base = ROOT / "data" / "docvqa"
    images_dir = base / "images"
    annotations_dir = base / "annotations"

    if images_dir.exists() and any(images_dir.iterdir()):
        print("  D-04 DocVQA: images/ already exists, skipping")
        return

    # Collect parquets from DocVQA/ subfolder
    parquet_dir = base / "DocVQA"
    parquets = sorted(parquet_dir.glob("*.parquet")) if parquet_dir.exists() else []
    if not parquets:
        print("  D-04 DocVQA: no parquet files found, skipping")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    seen_docs = set()
    count = 0
    for pq in parquets:
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            doc_id = f"{row['ucsf_document_id']}_{row['ucsf_document_page_no']}"
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            # Save image
            img_data = row["image"]
            if isinstance(img_data, dict) and img_data.get("bytes"):
                img = Image.open(io.BytesIO(img_data["bytes"]))
                img.save(images_dir / f"{doc_id}.png")
            # Save annotation -- DocVQA loader expects {"ocr_text": "..."}
            # We don't have OCR text in the QA dataset, so store the question+answers
            answers_raw = row.get("answers")
            answers = list(answers_raw) if answers_raw is not None else []
            answer_text = " | ".join(str(a) for a in answers) if answers else ""
            ann = {
                "question": str(row.get("question", "")),
                "answers": [str(a) for a in answers],
                "ocr_text": answer_text,
            }
            with open(annotations_dir / f"{doc_id}.json", "w") as f:
                json.dump(ann, f)
            count += 1

    print(f"  D-04 DocVQA: ready ({count} documents)")


# ---------------------------------------------------------------------------
# D-05  OlmOCR-Bench  (PDFs → images/ + ground_truth/ + tests/)
# ---------------------------------------------------------------------------

def prepare_olmocr_bench():
    base = ROOT / "data" / "olmocr_bench"
    images_dir = base / "images"
    tests_dir = base / "tests"
    gt_dir = base / "ground_truth"

    if images_dir.exists() and any(images_dir.iterdir()):
        print("  D-05 OlmOCR-Bench: images/ already exists, skipping")
        return

    bench_data = base / "bench_data"
    pdfs_dir = bench_data / "pdfs"
    if not pdfs_dir.exists():
        print("  D-05 OlmOCR-Bench: bench_data/pdfs/ not found, skipping")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Load all JSONL test entries, grouped by pdf
    pdf_tests: dict[str, list[dict]] = {}
    for jsonl_path in bench_data.glob("*.jsonl"):
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                pdf_path = entry.get("pdf", "")
                if pdf_path not in pdf_tests:
                    pdf_tests[pdf_path] = []
                pdf_tests[pdf_path].append(entry)

    # Render PDFs to images
    try:
        import pypdfium2 as pdfium
    except ImportError:
        print("  D-05 OlmOCR-Bench: pypdfium2 not installed, skipping")
        return

    count = 0
    for pdf_rel, tests in sorted(pdf_tests.items()):
        pdf_full = pdfs_dir / pdf_rel
        if not pdf_full.exists():
            continue

        doc_id = pdf_rel.replace("/", "_").replace(".pdf", "")
        doc_images_dir = images_dir / doc_id
        doc_images_dir.mkdir(parents=True, exist_ok=True)

        # Render each page referenced in the tests
        pages_needed = set()
        for t in tests:
            pages_needed.add(t.get("page", 1) - 1)  # 1-indexed → 0-indexed

        try:
            pdf_doc = pdfium.PdfDocument(str(pdf_full))
            for page_idx in sorted(pages_needed):
                if page_idx >= len(pdf_doc):
                    continue
                page = pdf_doc[page_idx]
                bitmap = page.render(scale=2)  # 144 DPI
                pil_img = bitmap.to_pil()
                pil_img.save(doc_images_dir / f"page_{page_idx}.png")
            pdf_doc.close()
        except Exception as e:
            print(f"    Warning: failed to render {pdf_rel}: {e}")
            continue

        # Save unit tests
        with open(tests_dir / f"{doc_id}.json", "w") as f:
            json.dump(tests, f)

        # Save ground truth (empty placeholder -- OlmOCR-Bench uses unit tests)
        gt = {}
        for t in tests:
            page_idx = t.get("page", 1) - 1
            gt[str(page_idx)] = t.get("math", t.get("text", ""))
        with open(gt_dir / f"{doc_id}.json", "w") as f:
            json.dump(gt, f)

        count += 1

    print(f"  D-05 OlmOCR-Bench: ready ({count} documents)")


# ---------------------------------------------------------------------------
# D-06  DocLayNet  (parquet → images/ + annotations/)
# ---------------------------------------------------------------------------

def prepare_doclaynet():
    base = ROOT / "data" / "doclaynet"
    images_dir = base / "images"
    annotations_dir = base / "annotations"

    if images_dir.exists() and any(images_dir.iterdir()):
        print("  D-06 DocLayNet: images/ already exists, skipping")
        return

    parquet_dir = base / "data"
    parquets = sorted(parquet_dir.glob("*.parquet")) if parquet_dir.exists() else []
    if not parquets:
        print("  D-06 DocLayNet: no parquet files found, skipping")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # DocLayNet category names (from the official spec)
    CATEGORY_NAMES = {
        1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item",
        5: "Page-footer", 6: "Page-header", 7: "Picture", 8: "Section-header",
        9: "Table", 10: "Text", 11: "Title",
    }

    count = 0
    for pq in parquets:
        df = pd.read_parquet(pq)
        for idx, row in df.iterrows():
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict):
                image_id = metadata.get("image_id", f"doclaynet_{count:06d}")
            else:
                image_id = f"doclaynet_{count:06d}"
            doc_id = str(image_id)

            doc_img_dir = images_dir / doc_id
            doc_img_dir.mkdir(parents=True, exist_ok=True)

            # Save image
            img_data = row["image"]
            if isinstance(img_data, dict) and img_data.get("bytes"):
                img = Image.open(io.BytesIO(img_data["bytes"]))
                img.save(doc_img_dir / "page_0.png")

            # Build COCO-style annotations
            bboxes = row.get("bboxes", [])
            categories = row.get("category_id", [])
            pdf_cells = row.get("pdf_cells", [])

            annotations = []
            # Extract text from pdf_cells
            texts_by_category = []
            if pdf_cells is not None:
                for i, cell_group in enumerate(pdf_cells):
                    if isinstance(cell_group, dict):
                        text = cell_group.get("text", "")
                        cat = int(categories[i]) if i < len(categories) else 0
                        texts_by_category.append({
                            "page_number": 0,
                            "text": text,
                            "category": CATEGORY_NAMES.get(cat, "Unknown"),
                        })
                    elif isinstance(cell_group, (list,)):
                        for cell in cell_group:
                            if isinstance(cell, dict):
                                text = cell.get("text", "")
                                cat = int(categories[i]) if i < len(categories) else 0
                                texts_by_category.append({
                                    "page_number": 0,
                                    "text": text,
                                    "category": CATEGORY_NAMES.get(cat, "Unknown"),
                                })

            ann = {"annotations": texts_by_category}
            with open(annotations_dir / f"{doc_id}.json", "w") as f:
                json.dump(ann, f)

            count += 1

    print(f"  D-06 DocLayNet: ready ({count} documents)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = {
    "D-02": ("FinTabNet", prepare_fintabnet),
    "D-03": ("FUNSD", prepare_funsd),
    "D-04": ("DocVQA", prepare_docvqa),
    "D-05": ("OlmOCR-Bench", prepare_olmocr_bench),
    "D-06": ("DocLayNet", prepare_doclaynet),
}


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    print("Preparing datasets...")
    for dataset_id in targets:
        if dataset_id not in DATASETS:
            print(f"  Unknown dataset: {dataset_id}")
            continue
        name, fn = DATASETS[dataset_id]
        print(f"\n  [{dataset_id}] {name}")
        fn()
    print("\nDone.")


if __name__ == "__main__":
    main()
