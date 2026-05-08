# tau3_airline

## Overview

An airline reservation management environment. Supports booking, modifying, and cancelling flight reservations, along with refunds and compensation via travel certificates.

The environment simulates a realistic airline system with flights across 20 US airports, user accounts with payment methods and membership tiers, and multi-segment reservations.

## Tools

### Query Tools (6)

| Tool | Description |
|------|-------------|
| `get_user_details` | Get user profile including reservations and payment methods |
| `get_reservation_details` | Get full reservation details |
| `get_flight_status` | Get status of a specific flight on a specific date |
| `search_direct_flight` | Search for direct flights between two cities on a date |
| `search_onestop_flight` | Search for one-stop flights between two cities on a date |
| `list_all_airports` | List all 20 available airports |

### Mutation Tools (8)

| Tool | Description |
|------|-------------|
| `book_reservation` | Book a new reservation with flights, passengers, payment |
| `cancel_reservation` | Cancel an entire reservation and issue refunds |
| `update_reservation_flights` | Change flights or cabin class of a reservation |
| `update_reservation_baggages` | Add checked bags to a reservation |
| `update_reservation_passengers` | Update passenger info (same count) |
| `send_certificate` | Issue a travel certificate to a user |
| `transfer_to_human_agents` | Transfer user to human agent with summary |
| `calculate` | Evaluate a mathematical expression |

## Database Schema

Three tables, each storing entities as JSON in a `data` column:

| Table | Primary Key | Rows | Description |
|-------|------------|------|-------------|
| `flights` | `flight_number` | 300 | Flight routes with per-date status and pricing |
| `users` | `user_id` | 500 | User profiles with payment methods and reservations |
| `reservations` | `reservation_id` | 2000 | Booking records with flights, passengers, payments |

### Key Data Structures

**Flight** (`flights.data`):
- `flight_number`, `origin`, `destination` (IATA codes)
- `scheduled_departure_time_est`, `scheduled_arrival_time_est`
- `dates`: dict of date -> status object (available/landed/cancelled/delayed)
- Available flights include `available_seats` and `prices` per cabin class

**User** (`users.data`):
- `user_id`, `name`, `email`, `dob`, `address`
- `membership`: regular / silver / gold
- `payment_methods`: dict of payment_id -> credit_card / gift_card / certificate
- `reservations`: list of reservation IDs

**Reservation** (`reservations.data`):
- `reservation_id`, `user_id`, `origin`, `destination`
- `flight_type`: one_way / round_trip
- `cabin`: basic_economy / economy / business
- `flights`: list of flight segments with prices
- `passengers`: list of passenger info
- `payment_history`: list of payments
- `insurance`: yes / no
- `status`: null (active) or "cancelled"

## Key Workflows

1. **Book**: authenticate user -> search flights -> collect passenger/payment info -> book_reservation
2. **Modify**: get user -> get reservation -> update flights/cabin/baggage/passengers
3. **Cancel**: get user -> get reservation -> verify cancellation eligibility -> cancel_reservation
4. **Compensate**: verify complaint -> send_certificate
