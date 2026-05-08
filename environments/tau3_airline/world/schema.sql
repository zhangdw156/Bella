CREATE TABLE flights (
    flight_number TEXT PRIMARY KEY,
    data TEXT NOT NULL
);

CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);

CREATE TABLE reservations (
    reservation_id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
