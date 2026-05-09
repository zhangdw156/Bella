"""Migrate BFCL v4 VehicleControlAPI multi-turn cases to Bella."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

BFCL_ROOT = Path("/Data/bywei/tmp/gorilla/berkeley-function-call-leaderboard")
BELLA_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = BELLA_ROOT / "cases"
ENV_NAME = "bfclv4_vehicle"
CATEGORY = "bfclv4_multi_base"
SOURCE = "bfcl"
CLASS_NAME = "VehicleControlAPI"

sys.path.insert(0, str(BFCL_ROOT))
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.vehicle_control import (
    DEFAULT_STATE,
    VehicleControlAPI,
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


def run_ground_truth(initial_config, ground_truth_turns):
    bot = VehicleControlAPI()
    bot._load_scenario(initial_config)
    for turn_calls in ground_truth_turns:
        for call_str in turn_calls:
            eval(f"bot.{call_str}")
    return bot


def config_to_world_setup(config):
    d = deepcopy(DEFAULT_STATE)
    stmts = []

    fuel = config.get("fuelLevel", d["fuelLevel"])
    batt = config.get("batteryVoltage", d["batteryVoltage"])
    engine = config.get("engineState", d["engine_state"])
    ac_temp = config.get("acTemperature", d["acTemperature"])
    fan = config.get("fanSpeed", d["fanSpeed"])
    ac_mode = config.get("acMode", d["acMode"])
    humidity = config.get("humidityLevel", d["humidityLevel"])
    headlight = config.get("headLightStatus", d["headLightStatus"])
    pbrake = config.get("parkingBrakeStatus", d["parkingBrakeStatus"])
    pbrake_force = config.get("parkingBrakeForce", d["_parkingBrakeForce"])
    slope = config.get("slopeAngle", d["_slopeAngle"])
    bpedal = config.get("brakePedalStatus", d["brakePedalStatus"])
    bpedal_force = config.get("brakePedalForce", d["brakePedalForce"])
    dist = config.get("distanceToNextVehicle", d["distanceToNextVehicle"])
    cruise = config.get("cruiseStatus", d["cruiseStatus"])
    dest = config.get("destination", d["destination"])
    fl = config.get("frontLeftTirePressure", d["frontLeftTirePressure"])
    fr = config.get("frontRightTirePressure", d["frontRightTirePressure"])
    rl = config.get("rearLeftTirePressure", d["rearLeftTirePressure"])
    rr = config.get("rearRightTirePressure", d["rearRightTirePressure"])
    seed = config.get("random_seed", d["random_seed"])

    door_status = config.get("doorStatus", d["doorStatus"])
    unlocked = 4 - len([1 for s in door_status.values() if s == "locked"])

    stmts.append(
        f"INSERT INTO vehicle_state VALUES ("
        f"{fuel}, {batt}, '{engine}', {unlocked}, "
        f"{ac_temp}, {fan}, '{ac_mode}', {humidity}, '{headlight}', "
        f"'{pbrake}', {pbrake_force}, {slope}, "
        f"'{bpedal}', {bpedal_force}, {dist}, '{cruise}', '{dest}', "
        f"{fl}, {fr}, {rl}, {rr}, {seed})"
    )

    for door_name, door_st in door_status.items():
        stmts.append(f"INSERT INTO door_status (door, status) VALUES ('{door_name}', '{door_st}')")

    return stmts


def generate_verify(config, bot_final):
    verify = []

    verify.append({
        "sql": "SELECT fuel_level FROM vehicle_state",
        "expected": [[bot_final.fuelLevel]],
        "order_matters": False,
    })
    verify.append({
        "sql": "SELECT engine_state FROM vehicle_state",
        "expected": [[bot_final.engine_state]],
        "order_matters": False,
    })
    verify.append({
        "sql": "SELECT remaining_unlocked_doors FROM vehicle_state",
        "expected": [[bot_final.remainingUnlockedDoors]],
        "order_matters": False,
    })

    door_status = bot_final.doorStatus
    init_doors = config.get("doorStatus", DEFAULT_STATE["doorStatus"])
    if door_status != init_doors:
        for door_name in ["driver", "passenger", "rear_left", "rear_right"]:
            verify.append({
                "sql": f"SELECT status FROM door_status WHERE door = '{door_name}'",
                "expected": [[door_status[door_name]]],
                "order_matters": False,
            })

    if bot_final.parkingBrakeStatus != config.get("parkingBrakeStatus", DEFAULT_STATE["parkingBrakeStatus"]):
        verify.append({
            "sql": "SELECT parking_brake_status FROM vehicle_state",
            "expected": [[bot_final.parkingBrakeStatus]],
            "order_matters": False,
        })

    if bot_final.brakePedalStatus != config.get("brakePedalStatus", DEFAULT_STATE["brakePedalStatus"]):
        verify.append({
            "sql": "SELECT brake_pedal_status, brake_pedal_force FROM vehicle_state",
            "expected": [[bot_final.brakePedalStatus, bot_final._brakePedalForce]],
            "order_matters": False,
        })

    init_dest = config.get("destination", DEFAULT_STATE["destination"])
    if bot_final.destination != init_dest:
        verify.append({
            "sql": "SELECT destination FROM vehicle_state",
            "expected": [[bot_final.destination]],
            "order_matters": False,
        })

    if bot_final.cruiseStatus != config.get("cruiseStatus", DEFAULT_STATE["cruiseStatus"]):
        verify.append({
            "sql": "SELECT cruise_status FROM vehicle_state",
            "expected": [[bot_final.cruiseStatus]],
            "order_matters": False,
        })

    if bot_final.headLightStatus != config.get("headLightStatus", DEFAULT_STATE["headLightStatus"]):
        verify.append({
            "sql": "SELECT head_light_status FROM vehicle_state",
            "expected": [[bot_final.headLightStatus]],
            "order_matters": False,
        })

    return verify


def extract_user_demands(question_turns):
    return [" ".join(m["content"] for m in turn if m.get("role") == "user") for turn in question_turns]


def main():
    cases, gt = load_bfcl_data()
    print(f"Found {len(cases)} {CLASS_NAME} single-env cases")

    for idx, case in enumerate(cases):
        case_id = f"bfclv4_vehicle_{idx:03d}"
        bfcl_id = case["id"]
        config = case["initial_config"][CLASS_NAME]
        gt_turns = gt.get(bfcl_id, [])

        bot = run_ground_truth(deepcopy(config), gt_turns)
        world_setup = config_to_world_setup(config)
        demands = extract_user_demands(case["question"])
        verify = generate_verify(config, bot)

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
