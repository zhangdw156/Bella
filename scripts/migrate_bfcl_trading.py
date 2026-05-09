"""Migrate BFCL v4 TradingBot multi-turn cases to Bella.

Reads BFCL case data and ground truth, executes ground truth calls on the
original TradingBot class to compute expected final state, then generates
Bella case JSONs with world_setup SQL and verify SQL.

Usage:
    python scripts/migrate_bfcl_trading.py
"""

from __future__ import annotations

import json
import re
import sys
from copy import deepcopy
from pathlib import Path

BFCL_ROOT = Path("/Data/bywei/tmp/gorilla/berkeley-function-call-leaderboard")
BELLA_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = BELLA_ROOT / "cases"
ENV_NAME = "bfclv4_trading"
CATEGORY = "bfclv4_multi_base"
SOURCE = "bfcl"

sys.path.insert(0, str(BFCL_ROOT))
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot import (
    TradingBot,
)


def load_bfcl_data() -> tuple[list[dict], dict[str, list[list[str]]]]:
    cases = []
    with open(BFCL_ROOT / "bfcl_eval/data/BFCL_v4_multi_turn_base.json") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("involved_classes") == ["TradingBot"]:
                cases.append(entry)

    gt = {}
    with open(
        BFCL_ROOT / "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json"
    ) as f:
        for line in f:
            entry = json.loads(line)
            gt[entry["id"]] = entry["ground_truth"]

    return cases, gt


def run_ground_truth(
    initial_config: dict, ground_truth_turns: list[list[str]]
) -> TradingBot:
    bot = TradingBot()
    bot._load_scenario(initial_config)
    for turn_calls in ground_truth_turns:
        for call_str in turn_calls:
            eval(f"bot.{call_str}")
    return bot


def initial_config_to_world_setup(config: dict) -> list[str]:
    stmts: list[str] = []

    acct = config.get("account_info", {})
    stmts.append(
        f"INSERT INTO trading_account (account_id, balance, binding_card) "
        f"VALUES ({acct.get('account_id', 12345)}, {acct.get('balance', 10000.0)}, {acct.get('binding_card', 0)})"
    )

    orders = config.get("orders", {})
    for oid_str, odata in orders.items():
        if not isinstance(odata, dict) or "symbol" not in odata:
            continue
        oid = int(oid_str) if str(oid_str).isdigit() else None
        if oid is None:
            continue
        otype = odata.get("order_type", orders.get("order_type", "Buy"))
        stmts.append(
            f"INSERT INTO orders (order_id, order_type, symbol, price, amount, status) "
            f"VALUES ({oid}, '{otype}', '{odata['symbol']}', {odata['price']}, "
            f"{odata.get('amount', odata.get('num_shares', 0))}, '{odata['status']}')"
        )

    stocks = config.get("stocks", {})
    for sym, sdata in stocks.items():
        stmts.append(
            f"INSERT INTO stocks (symbol, price, percent_change, volume, ma_5, ma_20) "
            f"VALUES ('{sym}', {sdata['price']}, {sdata['percent_change']}, "
            f"{sdata['volume']}, {sdata['MA(5)']}, {sdata['MA(20)']})"
        )

    for sym in config.get("watch_list", []):
        stmts.append(f"INSERT INTO watch_list (symbol) VALUES ('{sym}')")

    for tx in config.get("transaction_history", []):
        if "type" not in tx or "amount" not in tx:
            continue
        stmts.append(
            f"INSERT INTO transaction_history (type, amount, timestamp) "
            f"VALUES ('{tx['type']}', {tx['amount']}, '{tx['timestamp']}')"
        )

    auth = "1" if config.get("authenticated", False) else "0"
    stmts.append(
        f"INSERT INTO trading_state (key, value) VALUES ('authenticated', '{auth}')"
    )
    stmts.append(
        f"INSERT INTO trading_state (key, value) VALUES ('market_status', '{config.get('market_status', 'Closed')}')"
    )
    stmts.append(
        f"INSERT INTO trading_state (key, value) VALUES ('order_counter', '{config.get('order_counter', 12446)}')"
    )
    seed = config.get("random_seed", 1053520)
    stmts.append(
        f"INSERT INTO trading_state (key, value) VALUES ('random_seed', '{seed}')"
    )

    return stmts


def generate_verify(
    initial_config: dict, bot_final: TradingBot
) -> list[dict]:
    verify: list[dict] = []

    verify.append(
        {
            "sql": "SELECT balance FROM trading_account",
            "expected": [[bot_final.account_info["balance"]]],
            "order_matters": False,
        }
    )

    final_orders = bot_final.orders
    init_orders = initial_config.get("orders", {})
    for oid, odata in final_orders.items():
        if not isinstance(odata, dict) or "symbol" not in odata:
            continue
        oid_int = int(oid)
        init_status = None
        for k, v in init_orders.items():
            if str(k).isdigit() and int(k) == oid_int and isinstance(v, dict):
                init_status = v.get("status")
        if init_status != odata.get("status"):
            verify.append(
                {
                    "sql": f"SELECT status FROM orders WHERE order_id = {oid_int}",
                    "expected": [[odata["status"]]],
                    "order_matters": False,
                }
            )
        if init_status is None:
            verify.append(
                {
                    "sql": f"SELECT order_type, symbol, price, amount, status FROM orders WHERE order_id = {oid_int}",
                    "expected": [
                        [
                            odata["order_type"],
                            odata["symbol"],
                            odata["price"],
                            odata["amount"],
                            odata["status"],
                        ]
                    ],
                    "order_matters": False,
                }
            )

    final_wl = sorted(bot_final.watch_list)
    init_wl = sorted(initial_config.get("watch_list", []))
    if final_wl != init_wl:
        verify.append(
            {
                "sql": "SELECT symbol FROM watch_list ORDER BY symbol",
                "expected": [[s] for s in final_wl],
                "order_matters": True,
            }
        )

    if bot_final.authenticated != initial_config.get("authenticated", False):
        auth_val = 1 if bot_final.authenticated else 0
        verify.append(
            {
                "sql": "SELECT value FROM trading_state WHERE key = 'authenticated'",
                "expected": [[str(auth_val)]],
                "order_matters": False,
            }
        )

    init_tx_count = len(initial_config.get("transaction_history", []))
    final_tx_count = len(bot_final.transaction_history)
    if final_tx_count > init_tx_count:
        verify.append(
            {
                "sql": "SELECT COUNT(*) FROM transaction_history",
                "expected": [[final_tx_count]],
                "order_matters": False,
            }
        )
        for tx in bot_final.transaction_history[init_tx_count:]:
            verify.append(
                {
                    "sql": f"SELECT type, amount FROM transaction_history WHERE type = '{tx['type']}' AND amount = {tx['amount']}",
                    "expected": [[tx["type"], tx["amount"]]],
                    "order_matters": False,
                }
            )

    final_counter = bot_final.order_counter
    init_counter = initial_config.get("order_counter", 12446)
    if final_counter != init_counter:
        verify.append(
            {
                "sql": "SELECT value FROM trading_state WHERE key = 'order_counter'",
                "expected": [[str(final_counter)]],
                "order_matters": False,
            }
        )

    return verify


def extract_user_demands(question_turns: list[list[dict]]) -> list[str]:
    demands = []
    for turn in question_turns:
        parts = []
        for msg in turn:
            if msg.get("role") == "user":
                parts.append(msg["content"])
        demands.append(" ".join(parts))
    return demands


def main() -> None:
    cases, gt = load_bfcl_data()
    print(f"Found {len(cases)} TradingBot single-env cases")

    generated = 0
    for idx, case in enumerate(cases):
        case_id_str = f"bfclv4_trading_{idx:03d}"
        bfcl_id = case["id"]
        config = case["initial_config"]["TradingBot"]
        ground_truth_turns = gt.get(bfcl_id, [])

        bot_final = run_ground_truth(deepcopy(config), ground_truth_turns)

        world_setup = initial_config_to_world_setup(config)
        user_demands = extract_user_demands(case["question"])
        verify = generate_verify(config, bot_final)

        tags = [f"bfcl_id:{bfcl_id}", f"turns:{len(case['question'])}"]
        if case.get("excluded_function"):
            tags.append(f"excluded:{','.join(case['excluded_function'])}")

        bella_case = {
            "case_id": case_id_str,
            "env_name": ENV_NAME,
            "category": CATEGORY,
            "source": SOURCE,
            "tags": tags,
            "interaction_mode": "fixed",
            "user_demands": user_demands,
            "world_setup": world_setup,
            "verify": verify,
        }

        out_path = CASES_DIR / f"{case_id_str}.json"
        with open(out_path, "w") as f:
            json.dump(bella_case, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"  [{idx:03d}] {bfcl_id} -> {case_id_str} ({len(verify)} verify queries)")
        generated += 1

    print(f"\nGenerated {generated} cases in {CASES_DIR}")


if __name__ == "__main__":
    main()
