#!/usr/bin/env python3

import os
import sqlite3
import json
from datetime import datetime

# Configuration
BASE_DIR = os.getenv("BASE_DIR", ".")
CACHE_DB = os.getenv("CACHE_DB", os.path.join(BASE_DIR, "data", "prompt_cache.db"))
BACKUP_DB = os.path.join(BASE_DIR, f"backup_prompt_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")


def backup_database():
    """Creates a backup of the existing database before cleaning."""
    if os.path.exists(CACHE_DB):
        print(f"[INFO] Creating backup: {BACKUP_DB}")
        os.system(f"cp {CACHE_DB} {BACKUP_DB}")
    else:
        print("[WARNING] No database found to back up!")


def clean_database():
    """
    Cleans the SQLite database by removing:
      - Entries with '\"[LLM Error]\"' in response.
      - Entries with empty or null 'text' fields.
    """
    if not os.path.exists(CACHE_DB):
        print("[ERROR] Database file not found. Exiting.")
        return

    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()

    # Check total entries before cleanup
    cur.execute("SELECT COUNT(*) FROM prompt_cache")
    total_entries = cur.fetchone()[0]

    # Find corrupted entries
    cur.execute("SELECT id, prompt, response FROM prompt_cache")
    all_rows = cur.fetchall()

    delete_ids = []
    for row in all_rows:
        entry_id, prompt, response = row
        try:
            response_data = json.loads(response)
            # If 'text' is missing, empty, or "[LLM Error]", remove this entry
            if not response_data.get("text", "").strip() or response_data["text"].strip() == "[LLM Error]":
                delete_ids.append(entry_id)
        except json.JSONDecodeError:
            print(f"[WARNING] Invalid JSON for prompt '{prompt}' (ID {entry_id}) - Marking for deletion.")
            delete_ids.append(entry_id)

    # Delete invalid rows
    if delete_ids:
        print(f"[INFO] Deleting {len(delete_ids)} corrupted cache entries...")
        cur.executemany("DELETE FROM prompt_cache WHERE id = ?", [(entry_id,) for entry_id in delete_ids])
        conn.commit()
    else:
        print("[INFO] No corrupted entries found.")

    # Check total entries after cleanup
    cur.execute("SELECT COUNT(*) FROM prompt_cache")
    remaining_entries = cur.fetchone()[0]

    print(f"[INFO] Cleanup complete! {total_entries - remaining_entries} entries removed. {remaining_entries} valid entries remain.")

    conn.close()


if __name__ == "__main__":
    backup_database()
    clean_database()
