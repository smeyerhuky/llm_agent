# app.py

import sys
import signal
import logging
import json

from session_manager import SessionManager
from faiss_cache import (
    init_db, load_faiss, store_in_cache, top_k_cache_lookup,
    store_in_code_cache, top_k_code_cache_lookup, store_feedback
)
from docker_executor import force_cleanup_executor
from run_code import run_python_code_in_docker
from prompts import (
    classify_request, stream_llm_response, judge_response, finalize_task,
    FAST_MODEL, ADVANCED_MODEL, scrape_web
)
try:
    # If available
    from db_cleaner import clean_database
except ImportError:
    def clean_database():
        pass

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

# Session manager
session_manager = SessionManager()


def signal_handler(sig, frame):
    logger.info("Interrupt signal received. Forcing Docker cleanup & exiting.")
    force_cleanup_executor()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main_loop():
    """
    Main user-interactive loop that orchestrates the agent.
    """
    print("Welcome to the Improved LLM Agent!")
    print("Type 'exit' or 'quit' to end, or 'show context' to view session history.\n")

    # Initialize DB, run any cleaning, load FAISS
    init_db()
    logger.info("Running database cleanup...")
    clean_database()
    logger.info("DB cleanup complete!")

    load_faiss()

    # Add this line to ensure the execution service is running
    from docker_executor import ensure_execution_service
    ensure_execution_service()

    while True:
        user_prompt = input("[User] ").strip()
        if user_prompt.lower() in ("exit", "quit"):
            print("Exiting.")
            force_cleanup_executor()
            break
        if user_prompt.lower() == "show context":
            print("Session Context:")
            print(session_manager.dump_context())
            continue
        if user_prompt.lower().startswith("scrape:"):
            url = user_prompt.split("scrape:", 1)[1].strip()
            scraped_text = scrape_web(url)
            print("Scraped Content (truncated to 1k chars):")
            print(scraped_text[:1000], "...")
            session_manager[url] = scraped_text
            continue
        if "use module:" in user_prompt.lower():
            module_name = user_prompt.lower().split("use module:", 1)[1].strip()
            # For brevity, we won't re-implement module lookup logic here,
            # but you can do so as in your original code, store/lookup from module_cache, etc.
            print(f"Attempt to load module '{module_name}' from DB, import, etc.")
            continue

        # 1) Check top-k cache first
        cached_resp = top_k_cache_lookup(user_prompt)
        if cached_resp:
            logger.info("Cache Hit! Returning stored response.")
            print("### Cached Response")
            print(cached_resp)
            session_manager[user_prompt] = cached_resp

            action = input("Options: (r)erun code if it's code, (g)enerate new ignoring cache, or (i)gnore? (default=i): ").strip().lower() or "i"
            if action == "r" and cached_resp.get("type") == "code":
                # Re-run
                exec_result = run_python_code_in_docker(cached_resp["code"], user_prompt)
                print(exec_result["combined_output"])
                final_resp = cached_resp
            elif action == "g":
                # Generate new ignoring cache
                cat = classify_request(user_prompt)
                if cat == "code_needed":
                    messages = [
                        {"role": "system", "content": "You are an advanced Python developer. Output code in triple backticks if needed."},
                        {"role": "user", "content": f"Write a Python script for: {user_prompt}"}
                    ]
                    new_code = stream_llm_response(model=ADVANCED_MODEL, messages=messages)
                    exec_result = run_python_code_in_docker(new_code, user_prompt)
                    final_resp = {"type": "code", "code": new_code, "execution_result": exec_result, "code_path": exec_result["code_path"]}
                    store_in_cache(user_prompt, final_resp, ADVANCED_MODEL)
                    store_in_code_cache(user_prompt, final_resp, ADVANCED_MODEL)
                else:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant providing short text answers."},
                        {"role": "user", "content": user_prompt}
                    ]
                    new_text = stream_llm_response(model=FAST_MODEL, messages=messages)
                    final_resp = {"type": "text", "text": new_text}
                    store_in_cache(user_prompt, final_resp, FAST_MODEL)
            else:
                # i or default => use cached as-is
                final_resp = cached_resp

            # Judge / feedback / finalize
            ans_for_judge = final_resp.get("code", final_resp.get("text", ""))
            judgement = judge_response(user_prompt, ans_for_judge)
            print("\n[Judge's note]:", judgement)
            fb = input("Was this result good|bad|skip? (default=skip): ").strip().lower() or "skip"
            store_feedback(user_prompt, fb)

            code_path = final_resp.get("code_path", "N/A")
            if isinstance(final_resp.get("execution_result"), dict):
                summary_content = final_resp["execution_result"].get("result_summary", final_resp.get("text", ""))
            else:
                summary_content = final_resp.get("text", "")

            final_json = finalize_task(user_prompt, summary_content, code_path, judgement)
            print("\n[Final Output]:")
            print(final_json)
            continue

        # 2) Cache miss => we classify & proceed
        category = classify_request(user_prompt)
        final_resp = {}

        if category == "code_needed":
            # Check code_cache
            code_cache_hit = top_k_code_cache_lookup(user_prompt)
            if code_cache_hit:
                print("```")
                print(code_cache_hit["code"])
                print("```")
                if "execution_result" in code_cache_hit:
                    print("\n**Execution Result:**\n", code_cache_hit["execution_result"])

                act = input("Options: (r)erun, (g)enerate new code, or (i)gnore cache? (default=i): ").strip().lower() or "i"
                if act == "r":
                    exec_result = run_python_code_in_docker(code_cache_hit["code"], user_prompt)
                    final_resp = code_cache_hit
                elif act == "g":
                    # Regenerate
                    messages = [
                        {"role": "system", "content": "You are an advanced Python developer. Output code in triple backticks if needed."},
                        {"role": "user", "content": f"Write a Python script for: {user_prompt}"}
                    ]
                    code_str = stream_llm_response(model=ADVANCED_MODEL, messages=messages)
                    exec_result = run_python_code_in_docker(code_str, user_prompt)
                    final_resp = {"type": "code", "code": code_str, "execution_result": exec_result, "code_path": exec_result["code_path"]}
                    store_in_cache(user_prompt, final_resp, ADVANCED_MODEL)
                    store_in_code_cache(user_prompt, final_resp, ADVANCED_MODEL)
                else:
                    # i => ignore
                    final_resp = code_cache_hit

                # Judge
                judgement = judge_response(user_prompt, final_resp.get("code", ""))
                session_manager[user_prompt] = final_resp
                fb = input("Was this result good|bad|skip? ").strip().lower() or "skip"
                store_feedback(user_prompt, fb)
                continue

            # If no code cache hit, generate code from scratch
            logger.info("Classification => code_needed => generating new code with GPT-4.")
            messages = [
                {"role": "system", "content": "You are an advanced Python developer. Output code in triple backticks if needed."},
                {"role": "user", "content": f"Write a Python script for: {user_prompt}"}
            ]
            code_str = stream_llm_response(model=ADVANCED_MODEL, messages=messages)
            exec_result = run_python_code_in_docker(code_str, user_prompt)
            final_resp = {"type": "code", "code": code_str, "execution_result": exec_result, "code_path": exec_result["code_path"]}
            store_in_cache(user_prompt, final_resp, ADVANCED_MODEL)
            store_in_code_cache(user_prompt, final_resp, ADVANCED_MODEL)

            # Evaluate with judge
            judgement = judge_response(user_prompt, code_str)
            print("\n[Judge's note]:", judgement)
        else:
            # classification => no_code
            logger.info("Classification => no_code => using GPT-3.5 for text.")
            messages = [
                {"role": "system", "content": "You are a helpful assistant providing short text answers."},
                {"role": "user", "content": user_prompt}
            ]
            text_answer = stream_llm_response(model=FAST_MODEL, messages=messages)
            final_resp = {"type": "text", "text": text_answer}
            store_in_cache(user_prompt, final_resp, FAST_MODEL)

            judgement = judge_response(user_prompt, text_answer)
            print("\n[Judge's note]:", judgement)
            final_resp["code_path"] = "N/A"

        # At this point, final_resp is set with either code or text
        session_manager[user_prompt] = final_resp

        # Final pass => produce final JSON
        if final_resp.get("type") == "code":
            code_path = final_resp["execution_result"]["code_path"] if isinstance(final_resp.get("execution_result"), dict) else final_resp.get("code_path","N/A")
            result_summary = final_resp["execution_result"].get("result_summary", "")
        else:
            code_path = "N/A"
            result_summary = final_resp.get("text", "")


        final_json = finalize_task(user_prompt, result_summary, code_path, judgement)
        # Convert final_json to a pretty-printed JSON string
        formatted_final_json = json.dumps(final_json, indent=4)
        print("\n[Final Output]:")
        print(formatted_final_json)

        # Ask for feedback
        fb = input("Was this result good|bad|skip? (default=skip): ").strip().lower() or "skip"
        store_feedback(user_prompt, fb)

    # End of main loop


if __name__ == "__main__":
    main_loop()