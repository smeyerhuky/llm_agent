# faiss_cache.py

import os
import re
import json
import time
import sqlite3
import faiss
import numpy as np
import logging
from datetime import datetime
from typing import Optional, List, Tuple, Dict

from pydantic import BaseModel
from openai import OpenAI

logger = logging.getLogger(__name__)

# Expose these so the rest of the code can see the DB / index references easily
BASE_DIR = os.getenv("BASE_DIR", ".")
CACHE_DB = os.getenv("CACHE_DB", os.path.join(BASE_DIR, "data", "prompt_cache.db"))

# FAISS config
FAISS_DISTANCE_THRESHOLD = 0.2
FAISS_TOP_K = 3

# Initialize the global index (dimension=1536 for text-embedding-ada-002)
index = faiss.IndexFlatL2(1536)
stored_data: List[Tuple[str, int]] = []

# Create a single global OpenAI client for embeddings
embedding_client = OpenAI()

def init_db():
    """
    Create tables for caching, lessons, judge evaluations, and feedback if they don't exist.
    """
    db_dir = os.path.dirname(CACHE_DB)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()

    # Minimal table creation, add more if needed.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prompt_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT UNIQUE,
            response TEXT,
            embedding TEXT,
            model_version TEXT,
            timestamp DATETIME
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS code_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT UNIQUE,
            code TEXT,
            execution_result TEXT,
            embedding TEXT,
            model_version TEXT,
            timestamp DATETIME
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS judge_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT UNIQUE,
            judgement TEXT,
            timestamp DATETIME
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS lesson_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            lesson TEXT,
            timestamp DATETIME
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            feedback TEXT,
            timestamp DATETIME
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS module_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_name TEXT UNIQUE,
            file_path TEXT,
            code TEXT,
            timestamp DATETIME
        )
    """)

    conn.commit()
    conn.close()


def load_faiss():
    """
    Load embeddings from the prompt_cache into our global FAISS index.
    """
    global index, stored_data

    logger.debug("Loading FAISS index from DB...")
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("SELECT id, prompt, embedding FROM prompt_cache WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    conn.close()

    index.reset()
    stored_data.clear()

    embeddings = []
    for row_id, prompt_text, emb_json in rows:
        try:
            emb = json.loads(emb_json)
            if len(emb) == 1536:
                embeddings.append(emb)
                stored_data.append((prompt_text, row_id))
        except Exception as e:
            logger.warning(f"Parsing embedding for row {row_id} failed: {e}")

    if embeddings:
        index.add(np.array(embeddings, dtype=np.float32))
        logger.debug(f"FAISS index loaded with {len(embeddings)} embeddings.")
    else:
        logger.debug("No valid embeddings found in prompt_cache.")


def get_embedding(text: str) -> List[float]:
    """
    Return a 1536-dim embedding using text-embedding-ada-002
    """
    try:
        resp = embedding_client.embeddings.create(model="text-embedding-ada-002", input=text)
        emb = resp.data[0].embedding
        return emb
    except Exception as e:
        logger.warning(f"get_embedding failed for text[:30]={text[:30]}: {e}")
        return []


def store_in_cache(prompt: str, final_resp: dict, model_version: str):
    """
    Stores a user prompt, the final response, and the embedding in prompt_cache.
    Also updates the FAISS index in-memory.
    """
    emb = get_embedding(prompt)
    if len(emb) != 1536:
        logger.warning("Invalid embedding length, skipping store_in_cache.")
        return

    now_str = datetime.now().isoformat()
    emb_json = json.dumps(emb, default=str)
    resp_json = json.dumps(final_resp, default=str)

    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT OR REPLACE INTO prompt_cache (
                id, prompt, response, embedding, model_version, timestamp
            ) VALUES (
                COALESCE((SELECT id FROM prompt_cache WHERE prompt = ?), NULL),
                ?, ?, ?, ?, ?
            )
        """, (prompt, prompt, resp_json, emb_json, model_version, now_str))
        conn.commit()

        # Add to our in-memory store so we don't have to reload everything.
        new_id = cur.lastrowid
        stored_data.append((prompt, new_id))
        index.add(np.array([emb], dtype=np.float32))

        logger.debug("Stored new prompt cache entry in DB and updated FAISS index.")
    except Exception as e:
        logger.error(f"Failed to store prompt cache entry: {e}")
    finally:
        conn.close()


def top_k_cache_lookup(prompt: str) -> Optional[dict]:
    """
    Queries FAISS for similar prompts to 'prompt' in the prompt_cache.
    If we find one within FAISS_DISTANCE_THRESHOLD, returns the stored JSON response.
    Else None.
    """
    if index.ntotal == 0:
        return None

    emb = get_embedding(prompt)
    if len(emb) != 1536:
        return None

    query_vec = np.array([emb], dtype=np.float32)
    distances, ids = index.search(query_vec, k=FAISS_TOP_K)

    for rank, idx in enumerate(ids[0]):
        if distances[0][rank] < FAISS_DISTANCE_THRESHOLD:
            matched_prompt, row_id = stored_data[idx]
            logger.debug(f"FAISS match found at rank={rank}, distance={distances[0][rank]}, prompt={matched_prompt}.")
            conn = sqlite3.connect(CACHE_DB)
            cur = conn.cursor()
            row = cur.execute("SELECT response FROM prompt_cache WHERE id=?", (row_id,)).fetchone()
            conn.close()
            if row:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    pass
    return None


def store_in_code_cache(prompt: str, final_resp: dict, model_version: str):
    """
    If it's a code-based response, store it in code_cache with an embedding.
    """
    if final_resp.get("type") != "code":
        return

    emb = get_embedding(prompt)
    if len(emb) != 1536:
        logger.warning("Invalid embedding length, skipping code cache store.")
        return

    code_text = final_resp.get("code", "")
    exec_result = final_resp.get("execution_result", "")
    if not isinstance(exec_result, str):
        exec_result = json.dumps(exec_result, default=str)

    now_str = datetime.now().isoformat()
    emb_json = json.dumps(emb, default=str)

    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT OR REPLACE INTO code_cache (
                id, prompt, code, execution_result, embedding, model_version, timestamp
            ) VALUES (
                COALESCE((SELECT id FROM code_cache WHERE prompt=?), NULL),
                ?, ?, ?, ?, ?, ?
            )
        """, (prompt, prompt, code_text, exec_result, emb_json, model_version, now_str))
        conn.commit()
        logger.debug("Stored code response in code_cache.")
    except Exception as e:
        logger.error(f"Failed to store code cache entry: {e}")
    finally:
        conn.close()


def top_k_code_cache_lookup(prompt: str) -> Optional[dict]:
    """
    Look up similar prompts in code_cache. If within threshold, return code data.
    """
    emb = get_embedding(prompt)
    if len(emb) != 1536:
        return None

    best_match = None
    best_distance = float('inf')

    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    rows = cur.execute("SELECT prompt, code, execution_result, embedding FROM code_cache").fetchall()
    conn.close()

    for cached_prompt, code_text, exec_result, emb_json in rows:
        try:
            cached_emb = np.array(json.loads(emb_json), dtype=np.float32)
            distance = np.linalg.norm(np.array(emb, dtype=np.float32) - cached_emb)
            if distance < best_distance:
                best_distance = distance
                best_match = {"prompt": cached_prompt, "code": code_text, "execution_result": exec_result}
        except Exception as e:
            logger.warning(f"Error comparing embeddings in code_cache: {e}")

    if best_match and best_distance < FAISS_DISTANCE_THRESHOLD:
        logger.debug(f"Found code_cache match with distance={best_distance}")
        return best_match

    return None

def store_feedback(prompt: str, feedback: str):
    """
    Store user feedback (good, bad, skip) in feedback_cache.
    """
    now_str = datetime.now().isoformat()
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO feedback_cache (prompt, feedback, timestamp) VALUES (?, ?, ?)",
                    (prompt, feedback, now_str))
        conn.commit()
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
    finally:
        conn.close()

def store_judge_evaluation(prompt: str, judgement: dict):
    """
    Store the judge evaluation in judge_cache for reference.
    """
    now_str = datetime.now().isoformat()
    data_json = json.dumps(judgement, default=str)
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT OR REPLACE INTO judge_cache (prompt, judgement, timestamp)
            VALUES (?, ?, ?)
        """, (prompt, data_json, now_str))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to store judge evaluation: {e}")
    finally:
        conn.close()