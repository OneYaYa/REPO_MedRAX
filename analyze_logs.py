#!/usr/bin/env python
#conda run -n medrax python analyze_logs.py   qwen3-vl-4b-medrax-tools_20251226_191100.json
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

CHOICE_PATTERN = re.compile(r"\b([A-F])\b", re.IGNORECASE)


def extract_choice(answer_text: str) -> str:
    """从模型输出文本中解析出第一个 A-F 选项字母。"""
    if not isinstance(answer_text, str):
        return ""
    text = answer_text.strip()
    if text and text[0].upper() in "ABCDEF":
        return text[0].upper()
    m = CHOICE_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    return ""


def load_metadata(
    metadata_path: str,
    question_id_field: str = "question_id",
    categories_field: str = "categories",
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    读取 ChestAgentBench 的 metadata.jsonl。

    返回：
      - qid_to_categories: question_id -> [category_1, category_2, ...]
      - category_total: 每个类别的题目数量（一道题多类会被多次计数）
    """
    qid_to_categories: Dict[str, List[str]] = {}
    category_total: Dict[str, int] = defaultdict(int)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = record.get(question_id_field)
            if qid is None:
                continue
            qid = str(qid)

            raw_cats = record.get(categories_field, "")
            cats: List[str] = []

            # 你现在看到的是 "localization,comparison,relationship,reasoning" 这种字符串
            if isinstance(raw_cats, str):
                for c in raw_cats.split(","):
                    c = c.strip().lower()
                    if c:
                        cats.append(c)
            elif isinstance(raw_cats, list):
                for c in raw_cats:
                    if isinstance(c, str):
                        c = c.strip().lower()
                        if c:
                            cats.append(c)

            if not cats:
                continue

            qid_to_categories[qid] = cats
            for c in cats:
                category_total[c] += 1

    return qid_to_categories, category_total


def process_log_file(
    path: str,
    qid_to_categories: Dict[str, List[str]],
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    处理单个日志文件。

    返回：
      - global_stats: 不分类型的整体统计
      - category_stats: 按类别的统计；一道题如果属于多类，会更新多个类别的计数
    """
    global_stats = {
        "total_lines": 0,
        "answered": 0,
        "correct": 0,
        "incorrect": 0,
        "invalid_answer": 0,
        "skipped": 0,
        "error": 0,
    }

    category_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "total_lines": 0,
            "answered": 0,
            "correct": 0,
            "incorrect": 0,
            "invalid_answer": 0,
            "skipped": 0,
            "error": 0,
        }
    )

    if not os.path.exists(path):
        print(f"[WARN] Log file not found: {path}")
        return global_stats, category_stats

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            global_stats["total_lines"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Failed to parse JSON in {path}, line starts with: {line[:80]!r}")
                continue

            qid = str(record.get("question_id", "UNKNOWN"))
            cats = qid_to_categories.get(qid, ["unknown"])

            status = record.get("status", "").lower()

            # 每个类别都要有统计对象
            for cat in cats:
                _ = category_stats[cat]  # 触发 default

            if status == "skipped":
                global_stats["skipped"] += 1
                for cat in cats:
                    category_stats[cat]["skipped"] += 1
                    category_stats[cat]["total_lines"] += 1
                continue

            if status == "error":
                global_stats["error"] += 1
                for cat in cats:
                    category_stats[cat]["error"] += 1
                    category_stats[cat]["total_lines"] += 1
                continue

            model_answer = record.get("model_answer", None)
            correct_answer = record.get("correct_answer", None)

            if model_answer is None or correct_answer is None:
                # 不计入 answered，只记录有 status 的情况
                for cat in cats:
                    category_stats[cat]["total_lines"] += 1
                continue

            global_stats["answered"] += 1
            for cat in cats:
                category_stats[cat]["answered"] += 1
                category_stats[cat]["total_lines"] += 1

            pred_choice = extract_choice(model_answer)
            gold_choice = str(correct_answer).strip().upper()[:1]

            if pred_choice == "":
                global_stats["invalid_answer"] += 1
                global_stats["incorrect"] += 1
                for cat in cats:
                    category_stats[cat]["invalid_answer"] += 1
                    category_stats[cat]["incorrect"] += 1
                continue

            if pred_choice == gold_choice:
                global_stats["correct"] += 1
                for cat in cats:
                    category_stats[cat]["correct"] += 1
            else:
                global_stats["incorrect"] += 1
                for cat in cats:
                    category_stats[cat]["incorrect"] += 1

    return global_stats, category_stats


def merge_global_stats(all_globals: List[Dict[str, int]]) -> Dict[str, int]:
    merged = {
        "total_lines": 0,
        "answered": 0,
        "correct": 0,
        "incorrect": 0,
        "invalid_answer": 0,
        "skipped": 0,
        "error": 0,
    }
    for s in all_globals:
        for k in merged.keys():
            merged[k] += s.get(k, 0)
    return merged


def merge_category_stats(all_cats: List[Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, int]]:
    merged: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "total_lines": 0,
            "answered": 0,
            "correct": 0,
            "incorrect": 0,
            "invalid_answer": 0,
            "skipped": 0,
            "error": 0,
        }
    )
    for cat_stats in all_cats:
        for cat, s in cat_stats.items():
            ms = merged[cat]
            for k in ms.keys():
                ms[k] += s.get(k, 0)
    return merged


def safe_acc(correct: int, total: int) -> float:
    return (correct / total * 100.0) if total > 0 else 0.0


def _bucket_output_len(token_est: int) -> str:
    """
    把 tool_output_token_estimate 分桶，方便看趋势。
    你也可以按你数据分布改桶边界。
    """
    if token_est < 50:
        return "<50"
    if token_est < 100:
        return "50-99"
    if token_est < 200:
        return "100-199"
    if token_est < 400:
        return "200-399"
    if token_est < 800:
        return "400-799"
    return ">=800"


def process_log_file_tool_correlations(path: str) -> Dict[str, Dict]:
    """
    读取单个 log 文件，统计三类关系：
      1) accuracy vs num_tools_called
      2) accuracy vs tool_failures (count)
      3) accuracy vs tool_output_token_estimate (bucketed)

    注意：这里默认只统计 status=ok 且有 model_answer/correct_answer 的样本。
    """
    # 每个 bucket 都存 {n, answered, correct}
    by_num_tools: Dict[int, Dict[str, int]] = defaultdict(lambda: {"n": 0, "answered": 0, "correct": 0})
    by_failures: Dict[int, Dict[str, int]] = defaultdict(lambda: {"n": 0, "answered": 0, "correct": 0})
    by_outlen: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "answered": 0, "correct": 0})

    if not os.path.exists(path):
        print(f"[WARN] Log file not found: {path}")
        return {
            "by_num_tools": dict(by_num_tools),
            "by_failures": dict(by_failures),
            "by_outlen": dict(by_outlen),
        }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            status = str(record.get("status", "")).lower()
            if status != "ok":
                continue

            model_answer = record.get("model_answer", None)
            correct_answer = record.get("correct_answer", None)
            if model_answer is None or correct_answer is None:
                continue

            pred_choice = extract_choice(model_answer)
            gold_choice = str(correct_answer).strip().upper()[:1]
            is_correct = (pred_choice != "" and pred_choice == gold_choice)

            # 读你新增的三个字段；没有就给默认值，保证兼容旧 log
            num_tools = int(record.get("num_tools_called", 0) or 0)

            tf = record.get("tool_failures", {"count": 0})
            if isinstance(tf, dict):
                fail_count = int(tf.get("count", 0) or 0)
            else:
                # 兼容你未来万一写成纯 int 的情况
                fail_count = int(tf or 0)

            token_est = int(record.get("tool_output_token_estimate", 0) or 0)
            outlen_bucket = _bucket_output_len(token_est)

            # 更新 num_tools bucket
            by_num_tools[num_tools]["n"] += 1
            by_num_tools[num_tools]["answered"] += 1
            if is_correct:
                by_num_tools[num_tools]["correct"] += 1

            # 更新 failures bucket
            by_failures[fail_count]["n"] += 1
            by_failures[fail_count]["answered"] += 1
            if is_correct:
                by_failures[fail_count]["correct"] += 1

            # 更新 output length bucket
            by_outlen[outlen_bucket]["n"] += 1
            by_outlen[outlen_bucket]["answered"] += 1
            if is_correct:
                by_outlen[outlen_bucket]["correct"] += 1

    return {
        "by_num_tools": by_num_tools,
        "by_failures": by_failures,
        "by_outlen": by_outlen,
    }


def merge_bucket_stats_dict(
    dicts: List[Dict],
) -> Dict:
    """
    把多个文件的 bucket 统计合并。
    输入是 {bucket -> {n, answered, correct}} 结构（bucket 可以是 int 或 str）。
    """
    merged: Dict[Any, Dict[str, int]] = defaultdict(lambda: {"n": 0, "answered": 0, "correct": 0})
    for d in dicts:
        for bucket, s in d.items():
            ms = merged[bucket]
            ms["n"] += int(s.get("n", 0) or 0)
            ms["answered"] += int(s.get("answered", 0) or 0)
            ms["correct"] += int(s.get("correct", 0) or 0)
    return merged


def _print_bucket_table(title: str, bucket_stats: Dict, bucket_order: List = None):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'Bucket':>12s}{'N':>10s}{'Correct':>10s}{'Acc%':>10s}")
    print("-" * 42)

    items = list(bucket_stats.items())
    if bucket_order is not None:
        # 按用户指定顺序；没出现的桶跳过；出现但不在 order 的放后面
        order_index = {b: i for i, b in enumerate(bucket_order)}
        items.sort(key=lambda kv: order_index.get(kv[0], 10**9))
    else:
        # int bucket 走数值排序，str bucket 走字母排序
        if items and isinstance(items[0][0], int):
            items.sort(key=lambda kv: kv[0])
        else:
            items.sort(key=lambda kv: str(kv[0]))

    for bucket, s in items:
        n = int(s.get("answered", 0) or 0)
        c = int(s.get("correct", 0) or 0)
        acc = safe_acc(c, n)
        print(f"{str(bucket):>12s}{n:10d}{c:10d}{acc:10.2f}")

    print("=" * 80)


def print_summary(
    merged_global: Dict[str, int],
    merged_cat_stats: Dict[str, Dict[str, int]],
    category_total: Dict[str, int],
    treat_unanswered_as_wrong: bool = True,
    per_file_stats: List[Tuple[str, Dict[str, int]]] = None,
):
    print("=" * 80)
    print("ChestAgentBench evaluation summary (overall)")
    print("=" * 80)

    per_file_stats = False
    if per_file_stats:
        for path, s in per_file_stats:
            acc = safe_acc(s["correct"], s["answered"])
            print(f"\nFile: {os.path.basename(path)}")
            print(f"  Total log lines:        {s['total_lines']}")
            print(f"  Answered samples:       {s['answered']}")
            print(f"  Correct:                {s['correct']}")
            print(f"  Incorrect:              {s['incorrect']}")
            print(f"  Invalid answers:        {s['invalid_answer']}")
            print(f"  Skipped (status=skip):  {s['skipped']}")
            print(f"  Errors  (status=error): {s['error']}")
            print(f"  Accuracy (answered):    {acc:.2f}%")

    total_lines = merged_global["total_lines"]
    answered = merged_global["answered"]
    correct = merged_global["correct"]
    incorrect = merged_global["incorrect"]
    invalid_answer = merged_global["invalid_answer"]
    skipped = merged_global["skipped"]
    error = merged_global["error"]

    overall_acc_answered = safe_acc(correct, answered)

    print("\nOverall (all logs combined):")
    print(f"  Total log lines:        {total_lines}")
    print(f"  Answered samples:       {answered}")
    print(f"  Correct:                {correct}")
    print(f"  Incorrect:              {incorrect}")
    print(f"  Invalid answers:        {invalid_answer}")
    print(f"  Skipped (status=skip):  {skipped}")
    print(f"  Errors  (status=error): {error}")
    print(f"  Accuracy (answered):    {overall_acc_answered:.2f}%")

    print("\n" + "=" * 80)
    print("Per-category summary")
    print("=" * 80)

    # 按论文七大类优先排序，其它类别（如 reasoning）后面按字母序
    preferred_order = [
        "detection",
        "classification",
        "localization",
        "comparison",
        "relationship",
        "diagnosis",
        "characterization",
        "reasoning",
    ]
    cats_in_data = set(list(merged_cat_stats.keys()) + list(category_total.keys()))
    ordered_categories: List[str] = []
    for c in preferred_order:
        if c in cats_in_data:
            ordered_categories.append(c)
    for c in sorted(cats_in_data):
        if c not in ordered_categories:
            ordered_categories.append(c)

    header = (
        f"{'Category':18s}"
        f"{'TotalQ':>8s}"
        f"{'Answered':>10s}"
        f"{'Correct':>10s}"
        f"{'Acc(ans)%':>10s}"
        f"{'Acc(all)%':>10s}"
        f"{'Skipped':>10s}"
        f"{'Error':>8s}"
    )
    print(header)
    print("-" * len(header))

    cat_rows_for_latex = []

    for cat in ordered_categories:
        stats = merged_cat_stats.get(
            cat,
            {
                "total_lines": 0,
                "answered": 0,
                "correct": 0,
                "incorrect": 0,
                "invalid_answer": 0,
                "skipped": 0,
                "error": 0,
            },
        )
        total_q = category_total.get(cat, 0)
        answered = stats["answered"]
        correct = stats["correct"]
        skipped = stats["skipped"]
        error = stats["error"]

        acc_answered = safe_acc(correct, answered)
        if treat_unanswered_as_wrong:
            denom_all = total_q if total_q > 0 else answered
        else:
            denom_all = answered
        acc_all = safe_acc(correct, denom_all)

        display_cat = cat.capitalize()

        print(
            f"{display_cat:18s}"
            f"{total_q:8d}"
            f"{answered:10d}"
            f"{correct:10d}"
            f"{acc_answered:10.2f}"
            f"{acc_all:10.2f}"
            f"{skipped:10d}"
            f"{error:8d}"
        )

        cat_rows_for_latex.append((display_cat, total_q, acc_all))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MedRAX ChestAgentBench logs by category (multi-label categories)"
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        help="One or more JSON log files produced by quickstart.py",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="chestagentbench/metadata.jsonl",
        help="Path to metadata.jsonl from ChestAgentBench",
    )
    parser.add_argument(
        "--question-id-field",
        type=str,
        default="question_id",
        help="Field name for question id in metadata.jsonl",
    )
    parser.add_argument(
        "--categories-field",
        type=str,
        default="categories",
        help="Field name for categories in metadata.jsonl (comma-separated string)",
    )
    parser.add_argument(
        "--no-unanswered-as-wrong",
        action="store_true",
        help="If set, per-category accuracy is computed only on answered questions",
    )
    args = parser.parse_args()

    print(f"Loading metadata from: {args.metadata}")
    qid_to_categories, category_total = load_metadata(
        args.metadata,
        question_id_field=args.question_id_field,
        categories_field=args.categories_field,
    )

    per_file_global: List[Tuple[str, Dict[str, int]]] = []
    per_file_cat: List[Dict[str, Dict[str, int]]] = []

    for path in args.log_files:
        g_stats, c_stats = process_log_file(path, qid_to_categories)
        per_file_global.append((path, g_stats))
        per_file_cat.append(c_stats)
    
    # 新增：tool correlation 统计（每个文件一份）
    per_file_toolcorr = []
    for path in args.log_files:
        per_file_toolcorr.append(process_log_file_tool_correlations(path))


    merged_global = merge_global_stats([g for _, g in per_file_global])
    merged_cat = merge_category_stats(per_file_cat)

    treat_unanswered_as_wrong = not args.no_unanswered_as_wrong
    print_summary(
        merged_global,
        merged_cat,
        category_total,
        treat_unanswered_as_wrong=treat_unanswered_as_wrong,
        per_file_stats=per_file_global,
    )

    # 合并多个文件的 bucket 统计
    merged_by_num_tools = merge_bucket_stats_dict([d["by_num_tools"] for d in per_file_toolcorr])
    merged_by_failures = merge_bucket_stats_dict([d["by_failures"] for d in per_file_toolcorr])
    merged_by_outlen = merge_bucket_stats_dict([d["by_outlen"] for d in per_file_toolcorr])

    # 输出三张表
    _print_bucket_table(
        "Accuracy vs num_tools_called",
        merged_by_num_tools,
        bucket_order=None,
    )

    _print_bucket_table(
        "Accuracy vs tool_failures(count)",
        merged_by_failures,
        bucket_order=None,
    )

    # output length 的桶顺序固定一下更易读
    outlen_order = ["<50", "50-99", "100-199", "200-399", "400-799", ">=800"]
    _print_bucket_table(
        "Accuracy vs tool_output_token_estimate (bucketed)",
        merged_by_outlen,
        bucket_order=outlen_order,
    )



if __name__ == "__main__":
    main()
