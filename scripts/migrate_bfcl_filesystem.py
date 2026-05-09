"""Migrate BFCL v4 GorillaFileSystem multi-turn cases to Bella."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

BFCL_ROOT = Path("/Data/bywei/tmp/gorilla/berkeley-function-call-leaderboard")
BELLA_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = BELLA_ROOT / "cases"
ENV_NAME = "bfclv4_filesystem"
CATEGORY = "bfclv4_multi_base"
SOURCE = "bfcl"
CLASS_NAME = "GorillaFileSystem"

sys.path.insert(0, str(BFCL_ROOT))
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system import (
    Directory,
    File,
    GorillaFileSystem,
)


def load_bfcl_data():
    cases = []
    with open(BFCL_ROOT / "bfcl_eval/data/BFCL_v4_multi_turn_base.json") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("involved_classes") == [CLASS_NAME]:
                cases.append(entry)
    gt = {}
    with open(BFCL_ROOT / "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json") as f:
        for line in f:
            entry = json.loads(line)
            gt[entry["id"]] = entry["ground_truth"]
    return cases, gt


def run_ground_truth(initial_config, gt_turns):
    fs = GorillaFileSystem()
    fs._load_scenario(initial_config)
    for turn in gt_turns:
        for call_str in turn:
            eval(f"fs.{call_str}")
    return fs


def serialize_tree(directory, parent_path=None):
    """Convert GorillaFileSystem tree to flat (path, parent_path, name, is_dir, content) rows."""
    rows = []
    path = "" if parent_path is None else (f"{parent_path}/{directory.name}" if parent_path else directory.name)
    rows.append((path, parent_path, directory.name, 1, ""))
    for name, item in directory.contents.items():
        child_path = f"{path}/{name}" if path else name
        if isinstance(item, Directory):
            rows.extend(serialize_tree(item, path))
        elif isinstance(item, File):
            rows.append((child_path, path, name, 0, item.content))
    return rows


def rows_to_world_setup(rows):
    stmts = []
    for path, parent_path, name, is_dir, content in rows:
        p = "NULL" if parent_path is None else f"'{parent_path}'"
        content_escaped = content.replace("'", "''")
        stmts.append(
            f"INSERT INTO fs_entries (path, parent_path, name, is_directory, content) "
            f"VALUES ('{path}', {p}, '{name}', {is_dir}, '{content_escaped}')"
        )
    stmts.append("INSERT INTO fs_state (key, value) VALUES ('current_dir', '')")
    return stmts


def get_current_dir_path(fs):
    path_parts = []
    d = fs._current_dir
    while d is not None:
        path_parts.append(d.name)
        d = d.parent
    path_parts.reverse()
    if len(path_parts) <= 1:
        return ""
    return "/".join(path_parts[1:])


def generate_verify(initial_config, fs_initial, fs_final):
    verify = []

    init_rows = {r[0]: r for r in serialize_tree(fs_initial.root)}
    final_rows = {r[0]: r for r in serialize_tree(fs_final.root)}

    for path, row in final_rows.items():
        if path == "":
            continue
        _, _, name, is_dir, content = row
        if path not in init_rows:
            if is_dir:
                verify.append({
                    "sql": f"SELECT is_directory FROM fs_entries WHERE path = '{path}'",
                    "expected": [[1]],
                    "order_matters": False,
                })
            else:
                content_escaped = content.replace("'", "''")
                verify.append({
                    "sql": f"SELECT content FROM fs_entries WHERE path = '{path}'",
                    "expected": [[content]],
                    "order_matters": False,
                })
        elif not is_dir and init_rows[path][4] != content:
            verify.append({
                "sql": f"SELECT content FROM fs_entries WHERE path = '{path}'",
                "expected": [[content]],
                "order_matters": False,
            })

    for path in init_rows:
        if path == "":
            continue
        if path not in final_rows:
            verify.append({
                "sql": f"SELECT COUNT(*) FROM fs_entries WHERE path = '{path}'",
                "expected": [[0]],
                "order_matters": False,
            })

    final_dir = get_current_dir_path(fs_final)
    init_dir = get_current_dir_path(fs_initial)
    if final_dir != init_dir:
        verify.append({
            "sql": "SELECT value FROM fs_state WHERE key = 'current_dir'",
            "expected": [[final_dir]],
            "order_matters": False,
        })

    return verify


def extract_user_demands(question_turns):
    return [" ".join(m["content"] for m in turn if m.get("role") == "user") for turn in question_turns]


def main():
    cases, gt = load_bfcl_data()
    print(f"Found {len(cases)} {CLASS_NAME} single-env cases")

    for idx, case in enumerate(cases):
        case_id = f"bfclv4_filesystem_{idx:03d}"
        bfcl_id = case["id"]
        config = case["initial_config"][CLASS_NAME]
        gt_turns = gt.get(bfcl_id, [])

        fs_initial = GorillaFileSystem()
        fs_initial._load_scenario(deepcopy(config))

        fs_final = run_ground_truth(deepcopy(config), gt_turns)

        init_rows = serialize_tree(fs_initial.root)
        world_setup = rows_to_world_setup(init_rows)
        demands = extract_user_demands(case["question"])
        verify = generate_verify(config, fs_initial, fs_final)

        tags = [f"bfcl_id:{bfcl_id}", f"turns:{len(case['question'])}"]
        if case.get("excluded_function"):
            tags.append(f"excluded:{','.join(case['excluded_function'])}")

        bella_case = {
            "case_id": case_id,
            "env_name": ENV_NAME,
            "category": CATEGORY,
            "source": SOURCE,
            "tags": tags,
            "interaction_mode": "fixed",
            "user_demands": demands,
            "world_setup": world_setup,
            "verify": verify,
        }

        out = CASES_DIR / f"{case_id}.json"
        with open(out, "w") as f:
            json.dump(bella_case, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  [{idx:03d}] {bfcl_id} -> {case_id} ({len(verify)} verify)")

    print(f"\nGenerated {len(cases)} cases")


if __name__ == "__main__":
    main()
