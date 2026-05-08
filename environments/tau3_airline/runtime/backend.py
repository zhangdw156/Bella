"""SQLite-backed airline environment backend.

Loads flights, users, and reservations from SQLite JSON columns into
in-memory dicts.  Tool calls operate on in-memory state (preserving
the original tau3-bench AirlineTools logic), then sync modified
entities back to SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any


class EnvironmentBackend:
    """Airline reservation backend backed by a session-local SQLite database."""

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._flights: dict[str, dict] = {}
        self._users: dict[str, dict] = {}
        self._reservations: dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        for row in self._conn.execute("SELECT flight_number, data FROM flights"):
            self._flights[row[0]] = json.loads(row[1])
        for row in self._conn.execute("SELECT user_id, data FROM users"):
            self._users[row[0]] = json.loads(row[1])
        for row in self._conn.execute("SELECT reservation_id, data FROM reservations"):
            self._reservations[row[0]] = json.loads(row[1])

    def _sync_user(self, user_id: str) -> None:
        self._conn.execute(
            "UPDATE users SET data = ? WHERE user_id = ?",
            (json.dumps(self._users[user_id], ensure_ascii=False), user_id),
        )
        self._conn.commit()

    def _sync_reservation(self, reservation_id: str) -> None:
        data = json.dumps(self._reservations[reservation_id], ensure_ascii=False)
        existing = self._conn.execute(
            "SELECT 1 FROM reservations WHERE reservation_id = ?", (reservation_id,)
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE reservations SET data = ? WHERE reservation_id = ?",
                (data, reservation_id),
            )
        else:
            self._conn.execute(
                "INSERT INTO reservations VALUES (?, ?)", (reservation_id, data)
            )
        self._conn.commit()

    def _sync_flight(self, flight_number: str) -> None:
        self._conn.execute(
            "UPDATE flights SET data = ? WHERE flight_number = ?",
            (json.dumps(self._flights[flight_number], ensure_ascii=False), flight_number),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    _TOOL_NAMES = frozenset({
        "book_reservation",
        "calculate",
        "cancel_reservation",
        "get_flight_status",
        "get_reservation_details",
        "get_user_details",
        "list_all_airports",
        "search_direct_flight",
        "search_onestop_flight",
        "send_certificate",
        "transfer_to_human_agents",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    })

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._TOOL_NAMES:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            method = getattr(self, f"_tool_{tool_name}")
            result = method(**arguments)
            return result
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Helpers (ported from tau3 AirlineTools)
    # ------------------------------------------------------------------

    def _get_user(self, user_id: str) -> dict:
        if user_id not in self._users:
            raise ValueError(f"User {user_id} not found")
        return self._users[user_id]

    def _get_reservation(self, reservation_id: str) -> dict:
        if reservation_id not in self._reservations:
            raise ValueError(f"Reservation {reservation_id} not found")
        return self._reservations[reservation_id]

    def _get_flight(self, flight_number: str) -> dict:
        if flight_number not in self._flights:
            raise ValueError(f"Flight {flight_number} not found")
        return self._flights[flight_number]

    def _get_flight_instance(self, flight_number: str, date: str) -> dict:
        flight = self._get_flight(flight_number)
        if date not in flight["dates"]:
            raise ValueError(f"Flight {flight_number} not found on date {date}")
        return flight["dates"][date]

    def _get_new_reservation_id(self) -> str:
        for reservation_id in ["HATHAT", "HATHAU", "HATHAV"]:
            if reservation_id not in self._reservations:
                return reservation_id
        raise ValueError("Too many reservations")

    def _get_new_payment_id(self) -> list[int]:
        return [3221322, 3221323, 3221324]

    def _get_datetime(self) -> str:
        return "2024-05-15T15:00:00"

    def _search_direct_flight(
        self,
        date: str,
        origin: str | None = None,
        destination: str | None = None,
        leave_after: str | None = None,
    ) -> list[dict]:
        results = []
        for flight in self._flights.values():
            check = (
                (origin is None or flight["origin"] == origin)
                and (destination is None or flight["destination"] == destination)
                and (date in flight["dates"])
                and (flight["dates"][date]["status"] == "available")
                and (
                    leave_after is None
                    or flight["scheduled_departure_time_est"] >= leave_after
                )
            )
            if check:
                fd = flight["dates"][date]
                results.append({
                    "flight_number": flight["flight_number"],
                    "origin": flight["origin"],
                    "destination": flight["destination"],
                    "status": "available",
                    "scheduled_departure_time_est": flight["scheduled_departure_time_est"],
                    "scheduled_arrival_time_est": flight["scheduled_arrival_time_est"],
                    "available_seats": fd["available_seats"],
                    "prices": fd["prices"],
                })
        return results

    def _payment_for_update(self, user: dict, payment_id: str, total_price: int) -> dict | None:
        pm = user["payment_methods"]
        if payment_id not in pm:
            raise ValueError("Payment method not found")
        method = pm[payment_id]
        if method["source"] == "certificate":
            raise ValueError("Certificate cannot be used to update reservation")
        elif method["source"] == "gift_card" and method["amount"] < total_price:
            raise ValueError("Gift card balance is not enough")

        if method["source"] == "gift_card":
            method["amount"] -= total_price

        payment = None
        if total_price != 0:
            payment = {"payment_id": payment_id, "amount": total_price}
        return payment

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_book_reservation(
        self,
        user_id: str,
        origin: str,
        destination: str,
        flight_type: str,
        cabin: str,
        flights: list[dict],
        passengers: list[dict],
        payment_methods: list[dict],
        total_baggages: int,
        nonfree_baggages: int,
        insurance: str,
    ) -> dict:
        user = self._get_user(user_id)
        reservation_id = self._get_new_reservation_id()

        reservation = {
            "reservation_id": reservation_id,
            "user_id": user_id,
            "origin": origin,
            "destination": destination,
            "flight_type": flight_type,
            "cabin": cabin,
            "flights": [],
            "passengers": deepcopy(passengers),
            "payment_history": deepcopy(payment_methods),
            "created_at": self._get_datetime(),
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "insurance": insurance,
            "status": None,
        }

        total_price = 0
        all_flight_dates = []

        for fi in flights:
            fn = fi["flight_number"]
            flight = self._get_flight(fn)
            fd = self._get_flight_instance(fn, fi["date"])

            if fd["status"] != "available":
                raise ValueError(f"Flight {fn} not available on date {fi['date']}")
            if fd["available_seats"][cabin] < len(passengers):
                raise ValueError(f"Not enough seats on flight {fn}")

            price = fd["prices"][cabin]
            reservation["flights"].append({
                "flight_number": fn,
                "origin": flight["origin"],
                "destination": flight["destination"],
                "date": fi["date"],
                "price": price,
            })
            all_flight_dates.append((fn, fi["date"]))
            total_price += price * len(passengers)

        if insurance == "yes":
            total_price += 30 * len(passengers)
        total_price += 50 * nonfree_baggages

        for pm in payment_methods:
            pid = pm["payment_id"]
            amount = pm["amount"]
            if pid not in user["payment_methods"]:
                raise ValueError(f"Payment method {pid} not found")
            upm = user["payment_methods"][pid]
            if upm["source"] in {"gift_card", "certificate"} and upm["amount"] < amount:
                raise ValueError(f"Not enough balance in payment method {pid}")

        total_payment = sum(p["amount"] for p in payment_methods)
        if total_payment != total_price:
            raise ValueError(
                f"Payment amount does not add up, total price is {total_price}, but paid {total_payment}"
            )

        for pm in payment_methods:
            pid = pm["payment_id"]
            amount = pm["amount"]
            upm = user["payment_methods"][pid]
            if upm["source"] == "gift_card":
                upm["amount"] -= amount
            elif upm["source"] == "certificate":
                del user["payment_methods"][pid]

        for fn, date in all_flight_dates:
            self._flights[fn]["dates"][date]["available_seats"][cabin] -= len(passengers)
            self._sync_flight(fn)

        self._reservations[reservation_id] = reservation
        user["reservations"].append(reservation_id)
        self._sync_reservation(reservation_id)
        self._sync_user(user_id)
        return reservation

    def _tool_calculate(self, expression: str) -> str:
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    def _tool_cancel_reservation(self, reservation_id: str) -> dict:
        reservation = self._get_reservation(reservation_id)
        refunds = []
        for payment in reservation["payment_history"]:
            refunds.append({
                "payment_id": payment["payment_id"],
                "amount": -payment["amount"],
            })
        reservation["payment_history"].extend(refunds)
        reservation["status"] = "cancelled"
        self._sync_reservation(reservation_id)
        return reservation

    def _tool_get_flight_status(self, flight_number: str, date: str) -> str:
        return self._get_flight_instance(flight_number, date)["status"]

    def _tool_get_reservation_details(self, reservation_id: str) -> dict:
        return self._get_reservation(reservation_id)

    def _tool_get_user_details(self, user_id: str) -> dict:
        return self._get_user(user_id)

    def _tool_list_all_airports(self) -> list[dict]:
        return [
            {"iata": "SFO", "city": "San Francisco"},
            {"iata": "JFK", "city": "New York"},
            {"iata": "LAX", "city": "Los Angeles"},
            {"iata": "ORD", "city": "Chicago"},
            {"iata": "DFW", "city": "Dallas"},
            {"iata": "DEN", "city": "Denver"},
            {"iata": "SEA", "city": "Seattle"},
            {"iata": "ATL", "city": "Atlanta"},
            {"iata": "MIA", "city": "Miami"},
            {"iata": "BOS", "city": "Boston"},
            {"iata": "PHX", "city": "Phoenix"},
            {"iata": "IAH", "city": "Houston"},
            {"iata": "LAS", "city": "Las Vegas"},
            {"iata": "MCO", "city": "Orlando"},
            {"iata": "EWR", "city": "Newark"},
            {"iata": "CLT", "city": "Charlotte"},
            {"iata": "MSP", "city": "Minneapolis"},
            {"iata": "DTW", "city": "Detroit"},
            {"iata": "PHL", "city": "Philadelphia"},
            {"iata": "LGA", "city": "LaGuardia"},
        ]

    def _tool_search_direct_flight(self, origin: str, destination: str, date: str) -> list[dict]:
        return self._search_direct_flight(date=date, origin=origin, destination=destination)

    def _tool_search_onestop_flight(self, origin: str, destination: str, date: str) -> list[list[dict]]:
        results = []
        for r1 in self._search_direct_flight(date=date, origin=origin, destination=None):
            r1["date"] = date
            date2 = (
                f"2024-05-{int(date[-2:]) + 1}"
                if "+1" in r1["scheduled_arrival_time_est"]
                else date
            )
            for r2 in self._search_direct_flight(
                date=date2,
                origin=r1["destination"],
                destination=destination,
                leave_after=r1["scheduled_arrival_time_est"],
            ):
                r2["date"] = date2
                results.append([r1, r2])
        return results

    def _tool_send_certificate(self, user_id: str, amount: int) -> str:
        user = self._get_user(user_id)
        for payment_id in [f"certificate_{pid}" for pid in self._get_new_payment_id()]:
            if payment_id not in user["payment_methods"]:
                user["payment_methods"][payment_id] = {
                    "id": payment_id,
                    "amount": amount,
                    "source": "certificate",
                }
                self._sync_user(user_id)
                return f"Certificate {payment_id} added to user {user_id} with amount {amount}."
        raise ValueError("Too many certificates")

    def _tool_transfer_to_human_agents(self, summary: str) -> str:
        return "Transfer successful"

    def _tool_update_reservation_baggages(
        self,
        reservation_id: str,
        total_baggages: int,
        nonfree_baggages: int,
        payment_id: str,
    ) -> dict:
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation["user_id"])
        total_price = 50 * max(0, nonfree_baggages - reservation["nonfree_baggages"])
        payment = self._payment_for_update(user, payment_id, total_price)
        if payment is not None:
            reservation["payment_history"].append(payment)
        reservation["total_baggages"] = total_baggages
        reservation["nonfree_baggages"] = nonfree_baggages
        self._sync_reservation(reservation_id)
        self._sync_user(reservation["user_id"])
        return reservation

    def _tool_update_reservation_flights(
        self,
        reservation_id: str,
        cabin: str,
        flights: list[dict],
        payment_id: str,
    ) -> dict:
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation["user_id"])

        total_price = 0
        reservation_flights = []
        for fi in flights:
            matching = next(
                (
                    rf for rf in reservation["flights"]
                    if rf["flight_number"] == fi["flight_number"]
                    and rf["date"] == fi["date"]
                    and cabin == reservation["cabin"]
                ),
                None,
            )
            if matching:
                total_price += matching["price"] * len(reservation["passengers"])
                reservation_flights.append(matching)
                continue

            flight = self._get_flight(fi["flight_number"])
            fd = self._get_flight_instance(fi["flight_number"], fi["date"])
            if fd["status"] != "available":
                raise ValueError(f"Flight {fi['flight_number']} not available on date {fi['date']}")
            if fd["available_seats"][cabin] < len(reservation["passengers"]):
                raise ValueError(f"Not enough seats on flight {fi['flight_number']}")

            rf = {
                "flight_number": fi["flight_number"],
                "date": fi["date"],
                "price": fd["prices"][cabin],
                "origin": flight["origin"],
                "destination": flight["destination"],
            }
            total_price += rf["price"] * len(reservation["passengers"])
            reservation_flights.append(rf)

        total_price -= sum(f["price"] for f in reservation["flights"]) * len(reservation["passengers"])
        payment = self._payment_for_update(user, payment_id, total_price)
        if payment is not None:
            reservation["payment_history"].append(payment)

        reservation["flights"] = reservation_flights
        reservation["cabin"] = cabin
        self._sync_reservation(reservation_id)
        self._sync_user(reservation["user_id"])
        return reservation

    def _tool_update_reservation_passengers(
        self, reservation_id: str, passengers: list[dict]
    ) -> dict:
        reservation = self._get_reservation(reservation_id)
        if len(passengers) != len(reservation["passengers"]):
            raise ValueError("Number of passengers does not match")
        reservation["passengers"] = deepcopy(passengers)
        self._sync_reservation(reservation_id)
        return reservation
