#!/usr/bin/env python3
"""Migrate airline cases from tau2-bench to Bella tau3 format.

Only migrates cases with DB-mutating actions (verifiable via SQL).
"""

import json
import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), "../../tau2-bench/data/tau2/domains/airline/tasks.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../cases")

ALREADY_MIGRATED: set[str] = set()  # regenerate all cases
DB_MUTATING = {
    "cancel_reservation",
    "book_reservation",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
}
NEW_RESERVATION_IDS = ["HATHAT", "HATHAU", "HATHAV"]


def build_verify(task: dict) -> list[dict]:
    actions = task["evaluation_criteria"]["actions"]
    mutating = [a for a in actions if a["name"] in DB_MUTATING]
    verify = []
    book_count = 0

    for action in mutating:
        name = action["name"]
        args = action["arguments"]

        if name == "cancel_reservation":
            rid = args["reservation_id"]
            verify.append({
                "sql": f"SELECT json_extract(data, '$.status') FROM reservations WHERE reservation_id = '{rid}'",
                "expected": [["cancelled"]],
                "order_matters": False,
            })

        elif name == "book_reservation":
            new_id = NEW_RESERVATION_IDS[book_count]
            book_count += 1
            verify.append({
                "sql": f"SELECT COUNT(*) FROM reservations WHERE reservation_id = '{new_id}'",
                "expected": [[1]],
                "order_matters": False,
            })
            verify.append({
                "sql": (
                    f"SELECT json_extract(data, '$.origin'), json_extract(data, '$.destination'), "
                    f"json_extract(data, '$.cabin') FROM reservations WHERE reservation_id = '{new_id}'"
                ),
                "expected": [[args["origin"], args["destination"], args["cabin"]]],
                "order_matters": False,
            })

        elif name == "update_reservation_flights":
            rid = args["reservation_id"]
            cabin = args["cabin"]
            verify.append({
                "sql": f"SELECT json_extract(data, '$.cabin') FROM reservations WHERE reservation_id = '{rid}'",
                "expected": [[cabin]],
                "order_matters": False,
            })

        elif name == "update_reservation_passengers":
            rid = args["reservation_id"]
            passengers = args["passengers"]
            verify.append({
                "sql": (
                    f"SELECT json_array_length(json_extract(data, '$.passengers')) "
                    f"FROM reservations WHERE reservation_id = '{rid}'"
                ),
                "expected": [[len(passengers)]],
                "order_matters": False,
            })
            for i, p in enumerate(passengers):
                verify.append({
                    "sql": (
                        f"SELECT json_extract(data, '$.passengers[{i}].first_name'), "
                        f"json_extract(data, '$.passengers[{i}].last_name') "
                        f"FROM reservations WHERE reservation_id = '{rid}'"
                    ),
                    "expected": [[p["first_name"], p["last_name"]]],
                    "order_matters": False,
                })

        elif name == "update_reservation_baggages":
            rid = args["reservation_id"]
            verify.append({
                "sql": f"SELECT json_extract(data, '$.total_baggages') FROM reservations WHERE reservation_id = '{rid}'",
                "expected": [[args["total_baggages"]]],
                "order_matters": False,
            })
            verify.append({
                "sql": f"SELECT json_extract(data, '$.nonfree_baggages') FROM reservations WHERE reservation_id = '{rid}'",
                "expected": [[args["nonfree_baggages"]]],
                "order_matters": False,
            })

    return deduplicate_verify(verify)


def deduplicate_verify(verify: list[dict]) -> list[dict]:
    """Remove duplicate SQL checks (e.g. multiple update_flights on same reservation checking same cabin)."""
    seen = {}
    result = []
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

    knowledge = "The current time is 2024-05-15 15:00:00 EST.\n" + (instructions.get("known_info") or "")
    unknown = instructions.get("unknown_info")
    if unknown:
        knowledge += f"\nDoes not know: {unknown}"

    task_id = int(task["id"])
    return {
        "case_id": f"tau3_airline_{task_id:03d}",
        "env_name": "tau3_airline",
        "category": "tau3_airline",
        "source": "tau3",
        "tags": [],
        "interaction_mode": "dynamic",
        "demand": instructions["reason_for_call"],
        "world_setup": [],
        "user_agent_config": {
            "role": "A customer calling about an airline reservation",
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
        if task["id"] in ALREADY_MIGRATED:
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
