#!/usr/bin/env python3
"""Migrate retail cases from tau2-bench to Bella tau3 format.

Only migrates cases with DB-mutating actions (verifiable via SQL).
"""

import json
import os

SOURCE_PATH = os.path.join(os.path.dirname(__file__), "../../tau2-bench/data/tau2/domains/retail/tasks.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../cases")

DB_MUTATING = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "return_delivered_order_items",
    "modify_user_address",
}

SELECTED_IDS: set[str] | None = {"6","26","55","74","91","92","97","99","105"}  # None = all mutating tasks


def build_verify(task: dict) -> list[dict]:
    actions = task["evaluation_criteria"]["actions"]
    mutating = [a for a in actions if a["name"] in DB_MUTATING]
    verify: list[dict] = []

    for action in mutating:
        name = action["name"]
        args = action["arguments"]

        if name == "cancel_pending_order":
            oid = args["order_id"]
            verify.append({
                "sql": f"SELECT json_extract(data, '$.status') FROM orders WHERE order_id = '{oid}'",
                "expected": [["cancelled"]],
                "order_matters": False,
            })
            verify.append({
                "sql": f"SELECT json_extract(data, '$.cancel_reason') FROM orders WHERE order_id = '{oid}'",
                "expected": [[args["reason"]]],
                "order_matters": False,
            })

        elif name == "exchange_delivered_order_items":
            oid = args["order_id"]
            verify.append({
                "sql": f"SELECT json_extract(data, '$.status') FROM orders WHERE order_id = '{oid}'",
                "expected": [["exchange requested"]],
                "order_matters": False,
            })
            sorted_items = sorted(args["item_ids"])
            verify.append({
                "sql": f"SELECT json_extract(data, '$.exchange_items') FROM orders WHERE order_id = '{oid}'",
                "expected": [[json.dumps(sorted_items, separators=(",", ":"))]],
                "order_matters": False,
            })
            sorted_new = sorted(args["new_item_ids"])
            verify.append({
                "sql": f"SELECT json_extract(data, '$.exchange_new_items') FROM orders WHERE order_id = '{oid}'",
                "expected": [[json.dumps(sorted_new, separators=(",", ":"))]],
                "order_matters": False,
            })

        elif name == "return_delivered_order_items":
            oid = args["order_id"]
            verify.append({
                "sql": f"SELECT json_extract(data, '$.status') FROM orders WHERE order_id = '{oid}'",
                "expected": [["return requested"]],
                "order_matters": False,
            })
            sorted_items = sorted(args["item_ids"])
            verify.append({
                "sql": f"SELECT json_extract(data, '$.return_items') FROM orders WHERE order_id = '{oid}'",
                "expected": [[json.dumps(sorted_items, separators=(",", ":"))]],
                "order_matters": False,
            })

        elif name == "modify_pending_order_items":
            oid = args["order_id"]
            verify.append({
                "sql": f"SELECT json_extract(data, '$.status') FROM orders WHERE order_id = '{oid}'",
                "expected": [["pending (item modified)"]],
                "order_matters": False,
            })

        elif name == "modify_pending_order_address":
            oid = args["order_id"]
            verify.append({
                "sql": (
                    f"SELECT json_extract(data, '$.address.address1'), "
                    f"json_extract(data, '$.address.city'), "
                    f"json_extract(data, '$.address.zip') "
                    f"FROM orders WHERE order_id = '{oid}'"
                ),
                "expected": [[args["address1"], args["city"], args["zip"]]],
                "order_matters": False,
            })

        elif name == "modify_pending_order_payment":
            oid = args["order_id"]
            pmid = args["payment_method_id"]
            verify.append({
                "sql": (
                    f"SELECT json_extract(data, '$.payment_history[1].payment_method_id') "
                    f"FROM orders WHERE order_id = '{oid}'"
                ),
                "expected": [[pmid]],
                "order_matters": False,
            })

        elif name == "modify_user_address":
            uid = args["user_id"]
            verify.append({
                "sql": (
                    f"SELECT json_extract(data, '$.address.address1'), "
                    f"json_extract(data, '$.address.city'), "
                    f"json_extract(data, '$.address.zip') "
                    f"FROM users WHERE user_id = '{uid}'"
                ),
                "expected": [[args["address1"], args["city"], args["zip"]]],
                "order_matters": False,
            })

    return deduplicate_verify(verify)


def deduplicate_verify(verify: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    result: list[dict] = []
    for v in verify:
        sql = v["sql"]
        if sql in seen:
            seen[sql] = v
            for i, r in enumerate(result):
                if r["sql"] == sql:
                    result[i] = v
                    break
        else:
            seen[sql] = v
            result.append(v)
    return result


def convert_task(task: dict) -> dict:
    instructions = task["user_scenario"]["instructions"]

    knowledge = instructions.get("known_info") or ""
    unknown = instructions.get("unknown_info")
    if unknown:
        knowledge += f"\nDoes not know: {unknown}"

    task_id = int(task["id"])
    return {
        "case_id": f"tau3_retail_{task_id:03d}",
        "env_name": "tau3_retail",
        "category": "tau3_retail",
        "source": "tau3",
        "tags": [],
        "interaction_mode": "dynamic",
        "demand": instructions["reason_for_call"],
        "world_setup": [],
        "user_agent_config": {
            "role": "A customer contacting retail support",
            "personality": instructions.get("task_instructions") or "",
            "knowledge_boundary": knowledge,
        },
        "verify": build_verify(task),
    }


def main():
    with open(SOURCE_PATH) as f:
        tasks = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generated = []

    for task in tasks:
        if SELECTED_IDS is not None and task["id"] not in SELECTED_IDS:
            continue
        actions = task["evaluation_criteria"]["actions"]
        has_mutation = any(a["name"] in DB_MUTATING for a in actions)
        if not has_mutation:
            continue

        case = convert_task(task)
        filename = os.path.join(OUTPUT_DIR, f"{case['case_id']}.json")
        with open(filename, "w") as f:
            json.dump(case, f, indent=2, ensure_ascii=False)
            f.write("\n")
        generated.append(case["case_id"])
        print(f"  {case['case_id']}: {len(case['verify'])} verify queries")

    print(f"\nGenerated {len(generated)} cases")


if __name__ == "__main__":
    main()
