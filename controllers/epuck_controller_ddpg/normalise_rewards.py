import sqlite3
from math import sqrt
from typing import Dict, List, Tuple
import numpy as np


def connect_to_db(db_name: str) -> sqlite3.Connection:
    """
    Establish a database connection.
    """
    conn = sqlite3.connect(db_name)
    return conn


def execute_query(cur: sqlite3.Cursor, query: str, run_ids: List[str]) -> List[Tuple]:
    """
    Execute query on the database.
    """
    cur.execute(query, run_ids)
    data = cur.fetchall()
    return data


def fetch_data(cur: sqlite3.Cursor, run_ids: List[str]) -> List[Tuple]:
    """
    Fetch data for specified run IDs.
    """
    query = f"SELECT * FROM agent_data WHERE run_id in ({', '.join(['?' for _ in run_ids])})"
    return execute_query(cur, query, run_ids)


def organize_data_by_episode(data: List[Tuple]) -> Dict[str, List[Tuple]]:
    """
    Organize data by run_id and episode.
    """
    organized_data = {}
    for row in data:
        key = f"{row[1]}|{row[2]}"
        if key in organized_data:
            organized_data[key].append(row)
        else:
            organized_data[key] = [row]
    return organized_data


def calculate_euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate Euclidean distance.
    """
    return np.linalg.norm(np.subtract([x1, y1], [x2, y2])) / sqrt(4**2 + 4**2)


def normalize_rewards(organized_data: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
    """
    Normalize episode rewards by Euclidean distance.
    """
    normalized_data = {}
    for key, episode in organized_data.items():
        final_x, final_y = episode[-1][9], episode[-1][10]
        for step in episode:
            distance = calculate_euclidean_distance(step[9], step[10], final_x, final_y)
            normalized_data.setdefault(key, []).append(step[4] / distance if distance else 0)
    return normalized_data


def arrange_by_run(normalized_data: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
    """
    Arrange data by run_id.
    """
    arranged_data = {}
    for key, episode in normalized_data.items():
        run_id = key.split("|")[0]
        arranged_data.setdefault(run_id, []).append(episode)
    return arranged_data

