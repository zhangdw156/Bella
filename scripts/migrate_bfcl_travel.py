"""Migrate BFCL v4 TravelAPI multi-turn cases to Bella."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

BFCL_ROOT = Path("/Data/bywei/tmp/gorilla/berkeley-function-call-leaderboard")
BELLA_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = BELLA_ROOT / "cases"
ENV_NAME = "bfclv4_travel"
CATEGORY = "bfclv4_multi_base"
SOURCE = "bfcl"
CLASS_NAME = "TravelAPI"

sys.path.insert(0, str(BFCL_ROOT))
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.travel_booking import (
    DEFAULT_STATE,
    TravelAPI,
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


def run_ground_truth(config, gt_turns):
    api = TravelAPI()
    api._load_scenario(config)
    for turn in gt_turns:
        for call_str in turn:
            eval(f"api.{call_str}")
    return api


def config_to_world_setup(config):
    stmts = []

    for card_id, card in config.get("credit_card_list", {}).items():
        cn = card.get("card_number", "")
        ed = card.get("expiration_date", "")
        ch = card.get("cardholder_name", "").replace("'", "''")
        cvv = card.get("card_verification_number", 0)
        bal = card.get("balance", 0)
        stmts.append(
            f"INSERT INTO credit_cards VALUES ('{card_id}', '{cn}', '{ed}', '{ch}', {cvv}, {bal})"
        )

    for bid, booking in config.get("booking_record", {}).items():
        cid = booking.get("card_id", "")
        td = booking.get("travel_date", "")
        tf = booking.get("travel_from", "")
        tt = booking.get("travel_to", "")
        tc = booking.get("travel_class", "")
        cost = booking.get("travel_cost", 0)
        tid = booking.get("transaction_id", "")
        stmts.append(
            f"INSERT INTO bookings VALUES ('{bid}', '{cid}', '{td}', '{tf}', '{tt}', '{tc}', {cost}, '{tid}')"
        )

    state_fields = [
        ("access_token", config.get("access_token")),
        ("token_type", config.get("token_type")),
        ("token_expires_in", str(config["token_expires_in"]) if config.get("token_expires_in") is not None else None),
        ("token_scope", config.get("token_scope")),
        ("user_first_name", config.get("user_first_name")),
        ("user_last_name", config.get("user_last_name")),
        ("budget_limit", str(config["budget_limit"]) if config.get("budget_limit") is not None else None),
        ("random_seed", str(config.get("random_seed", DEFAULT_STATE["random_seed"]))),
    ]
    for key, val in state_fields:
        if val is not None:
            stmts.append(f"INSERT INTO travel_state (key, value) VALUES ('{key}', '{val}')")
        else:
            stmts.append(f"INSERT INTO travel_state (key, value) VALUES ('{key}', NULL)")

    return stmts


def generate_verify(config, api_final):
    verify = []

    for card_id, card in api_final.credit_card_list.items():
        init_cards = config.get("credit_card_list", {})
        init_balance = init_cards.get(card_id, {}).get("balance")
        if init_balance != card.get("balance"):
            verify.append({
                "sql": f"SELECT balance FROM credit_cards WHERE card_id = '{card_id}'",
                "expected": [[card["balance"]]],
                "order_matters": False,
            })
        if card_id not in init_cards:
            verify.append({
                "sql": f"SELECT COUNT(*) FROM credit_cards WHERE card_id = '{card_id}'",
                "expected": [[1]],
                "order_matters": False,
            })

    for bid, booking in api_final.booking_record.items():
        init_bookings = config.get("booking_record", {})
        if bid not in init_bookings:
            verify.append({
                "sql": f"SELECT travel_from, travel_to, travel_class, travel_cost FROM bookings WHERE booking_id = '{bid}'",
                "expected": [[booking["travel_from"], booking["travel_to"], booking["travel_class"], booking["travel_cost"]]],
                "order_matters": False,
            })

    init_bookings = config.get("booking_record", {})
    for bid in init_bookings:
        if bid not in api_final.booking_record:
            verify.append({
                "sql": f"SELECT COUNT(*) FROM bookings WHERE booking_id = '{bid}'",
                "expected": [[0]],
                "order_matters": False,
            })

    if api_final.budget_limit != config.get("budget_limit"):
        val = str(api_final.budget_limit) if api_final.budget_limit is not None else None
        if val is not None:
            verify.append({
                "sql": "SELECT value FROM travel_state WHERE key = 'budget_limit'",
                "expected": [[val]],
                "order_matters": False,
            })

    init_booking_count = len(config.get("booking_record", {}))
    final_booking_count = len(api_final.booking_record)
    if final_booking_count != init_booking_count:
        verify.append({
            "sql": "SELECT COUNT(*) FROM bookings",
            "expected": [[final_booking_count]],
            "order_matters": False,
        })

    return verify


def extract_user_demands(question_turns):
    return [" ".join(m["content"] for m in turn if m.get("role") == "user") for turn in question_turns]


def main():
    cases, gt = load_bfcl_data()
    print(f"Found {len(cases)} {CLASS_NAME} single-env cases")

    for idx, case in enumerate(cases):
        case_id = f"bfclv4_travel_{idx:03d}"
        bfcl_id = case["id"]
        config = case["initial_config"][CLASS_NAME]
        gt_turns = gt.get(bfcl_id, [])

        api_final = run_ground_truth(deepcopy(config), gt_turns)
        world_setup = config_to_world_setup(config)
        demands = extract_user_demands(case["question"])
        verify = generate_verify(config, api_final)

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
