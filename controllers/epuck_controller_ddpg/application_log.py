# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     12 May 2023
#
# Purpose:  Provide a simple database logger for the application
#
# Schema is:
"""
CREATE TABLE app_log (
id INTEGER PRIMARY KEY,
timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
level TEXT,
function TEXT,
message TEXT);
"""
import os
import sqlite3
from sqlite3 import OperationalError


def log(level: str, function: str, message: str):
    """Log to the database. Level is one of DEBUG, INFO, WARNING, ERROR, CRITICAL."""
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(f"Invalid log level: {level}")

    conn = sqlite3.connect("/home/levinin/Documents/thesis/logs/thesis_app_log.db")
    cursor: sqlite3.Cursor = conn.cursor()

    while True:  # Keep trying until successful, otherwise may occasionally fail due to database lock
        try:
            cursor.execute("INSERT INTO app_log (id, level, function, message) VALUES (NULL, ?, ?, ?)",
                           (level, function, message))
            conn.commit()
            break
        except OperationalError as e:
            if "SQLITE_BUSY" not in str(e):
                raise e

    conn.close()
