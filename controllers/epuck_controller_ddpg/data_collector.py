# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     11 May 2023
# Purpose:  Record data to a database for later retrieval and analysis
#
# Notes:    The intention is to share run_id and episode between processes to track these together.
#           The structure allows aggregation at the run and episode level, and also allows for
#           analysis of the data at the step level and over time.
#
#           SQLite stores data as binary so is "compressed" vs csv. However, it can become fragmented, so it may
#           be good to use VACUUM and REINDEX to compact the database if fragmentation causes bloat.
"""
sqlite> .schema

CREATE TABLE agent_data (
id INTEGER PRIMARY KEY,
run_id TEXT,
episode INTEGER,
step INTEGER,
step_reward FLOAT,
episode_end BOOL,
goal_achieved BOOL,
timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
run_type TEXT,
x FLOAT,
y FLOAT);

CREATE TABLE optimise_data (
id INTEGER PRIMARY KEY,
run_id TEXT,
step INTEGER,
q_loss FLOAT,
a_loss FLOAT,
inter FLOAT,
q_val FLOAT,
episode INTEGER,
timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);
"""
import inspect
import sqlite3
from sqlite3 import OperationalError
from dataclasses import dataclass
from contextlib import contextmanager
from time import sleep

from torch.multiprocessing import Queue
import application_log

database_path = '../logs/data_logs.db'


@dataclass
class AgentData:
    run_id: str = ''
    episode: int = 0
    step: int = 0
    step_reward: float = 0.0
    episode_end: bool = False
    goal_achieved: bool = False
    location: list = None
    run_type: str = ''
    x: float = 0.0
    y: float = 0.0


@dataclass
class OptimiseData:
    run_id: str = ''
    step: int = 0
    q_loss: float = 0.0
    a_loss: float = 0.0
    inter: float = 0.0
    q_val: float = 0.0
    episode: int = 0


def write_agent_data(agent_data: AgentData, cursor: sqlite3.Cursor):
    cursor.execute("INSERT INTO agent_data (id, run_id, episode, step, "
                   "step_reward, episode_end, goal_achieved, run_type, x, y) "
                   "VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (agent_data.run_id, agent_data.episode, agent_data.step, agent_data.step_reward,
                    agent_data.episode_end, agent_data.goal_achieved, agent_data.run_type, agent_data.x, agent_data.y))


def write_optimise_data(optimise_data: OptimiseData, cursor: sqlite3.Cursor):
    cursor.execute("INSERT INTO optimise_data (id, run_id, step, q_loss, a_loss, inter, q_val, episode) "
                   "VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
                   (optimise_data.run_id, optimise_data.step, optimise_data.q_loss, optimise_data.a_loss,
                    optimise_data.inter, optimise_data.q_val, optimise_data.episode))


def agent_data_database_writer(queue: Queue):
    """Receives data through the queue and writes it to the database.
    Data is [<status>, <data>], where <status> is True or False and <data> is an AgentData object."""

    conn: sqlite3.Connection = sqlite3.connect(database_path)
    cursor: sqlite3.Cursor = conn.cursor()

    status: bool
    data: AgentData

    application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                        message="Agent data database writer started.")

    while True:                     # Keep going until told to stop
        status, data = queue.get()
        if status is False:
            application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                                message="Agent data database process received signal to stop.")
            break
        while True:                 # Keep trying until successful, otherwise may occasionally fail due to database lock
            try:
                write_agent_data(data, cursor)
                conn.commit()
                break
            except OperationalError as e:
                write_issue_check(e)

    cursor.execute("PRAGMA analysis_limit = 1000;")             # Optimise the database, recommended by sqlite docs
    cursor.execute("PRAGMA optimize;")
    conn.close()
    print("Agent data database writer stopped.")


def optimiser_data_database_writer(queue: Queue):
    """Receives data through the queue and writes it to the database.
    Data is [<status>, <data>], where <status> is True or False and <data> is an OptimiseData object."""

    conn: sqlite3.Connection = sqlite3.connect(database_path)
    cursor: sqlite3.Cursor = conn.cursor()

    status: bool
    data: OptimiseData

    application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                        message="Optimiser data database writer started.")

    while True:
        status, data = queue.get()
        if status is False:
            application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                                message="Optimiser database process received signal to stop.")
            break
        while True:  # Keep trying until successful, otherwise may occasionally fail due to database lock
            try:
                write_optimise_data(data, cursor)
                conn.commit()
                break
            except OperationalError as e:
                write_issue_check(e)

    conn.close()
    print("Optimiser data database writer stopped.")


def write_issue_check(e):
    """Check for write issues and raise exception if found"""
    if "SQLITE_BUSY" in str(e):
        application_log.log(level='WARNING', function=f"{inspect.stack()[0][3]}",
                            message=f"Write error: {e}")
    else:
        application_log.log(level='CRITICAL', function=f"{inspect.stack()[0][3]}",
                            message=f"Write error: {e}")
        raise e
