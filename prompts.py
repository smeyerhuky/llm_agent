# prompts.py

import json
import logging
import time
import re
import requests
from bs4 import BeautifulSoup
from typing import Optional
from pydantic import BaseModel, Field

from openai import OpenAI

from faiss_cache import store_judge_evaluation
from faiss_cache import store_feedback  # or you can keep that separate
from docker_executor import BASE_DIR

logger = logging.getLogger(__name__)

FAST_MODEL = "gpt-3.5-turbo"
ADVANCED_MODEL = "gpt-4"
FINAL_MODEL = "gpt-4o-mini"

client = OpenAI()

class Classification(BaseModel):
    category: str = Field(..., description="Either 'code_needed' or 'no_code'")

def classify_request(user_prompt: str) -> str:
    """
    Classify a user prompt to determine if code is needed, returning 'code_needed' or 'no_code'.
    """
    system_content = "You are a classification assistant. Output JSON with one field 'category', either 'code_needed' or 'no_code'."
    user_content = f"User prompt: {user_prompt}"

    try:
        resp = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            max_tokens=50,
            temperature=0.0,
        )
        raw_text = resp.choices[0].message.content.strip()
        parsed = json.loads(raw_text)
        classification = Classification(**parsed)
        return classification.category
    except Exception as e:
        logger.warning(f"Classification error => defaulting to 'no_code': {e}")
        return "no_code"

def generate_refined_prompt(user_prompt: str, code_str: str, execution_result: dict, attempt: int) -> list:
    """
    Generate an improved prompt for code regeneration based on execution errors.
    
    Args:
        user_prompt: The original user request
        code_str: The previous code that failed
        execution_result: The execution result containing error information
        attempt: The current retry attempt number (1-3)
    
    Returns:
        List of message dicts for the LLM
    """
    # Extract error information from execution result
    combined_output = execution_result.get("combined_output", "")
    error_summary = execution_result.get("result_summary", "")
    
    # Create a prompt that focuses on specific errors and includes previous code
    system_content = """You are an expert Python troubleshooter and developer. Your task is to fix code that failed execution.
Analyze the error messages carefully and produce a more robust solution that addresses these specific issues.
Your solution should include:
1. Error handling with try/except blocks for risky operations
2. Input validation where appropriate
3. Clear comments explaining your error-fixing approach
4. Only complete, standalone solutions (no fragments)

Focus especially on fixing the specific error without changing the core solution approach unless absolutely necessary."""

    # Adjust the prompt based on retry attempt
    if attempt == 1:
        focus = "Focus on the immediate errors shown in the output. Add basic error handling."
    elif attempt == 2:
        focus = "Take a more comprehensive approach. Address the specific errors but also anticipate related issues. Add robust error handling and input validation."
    else:  # attempt == 3
        focus = "This is the final retry attempt. Take a completely different approach if needed. Prioritize a working solution over elegance."
    
    user_content = f"""Original request: {user_prompt}

Previous code attempt:
```python
{code_str}
```

Execution output:
{combined_output}

Error summary: {error_summary}

{focus}

Generate a new, complete Python solution that fixes these issues."""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

def scrape_web(url: str) -> str:
    """
    Scrape a webpage and return text content (for demonstration).
    """
    logger.info(f"Scraping URL: {url}")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator="\n")
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return f"[Scraping Error] {e}"


def extract_code_from_chunk(chunk: str) -> str:
    """
    Extract code between triple backticks. If none found, return original chunk.
    """
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, chunk, flags=re.DOTALL)
    if matches:
        return "\n".join(m.strip() for m in matches)
    return chunk


def summarize_output(execution_output: str) -> str:
    """
    Summarize the Docker execution output into key lessons & a concise result summary.
    """
    prompt = (f"Summarize the following docker execution output into "
              f"key lessons and a concise result summary:\n\n{execution_output}\n\nSummary:")

    try:
        start_time = time.time()
        resp = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": "You are a results summarizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        summary = resp.choices[0].message.content.strip()
        duration_ms = int((time.time() - start_time) * 1000)
        logger.debug(f"Summarize took {duration_ms}ms")
        return summary
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return "Summary not available."

def summarize_refinement(old_code: str, docker_output: str) -> str:
    """
    Provide a brief summary of changes that might fix errors from the original code.
    """
    prompt = (
        f"The following Docker execution produced errors:\n{docker_output}\n\n"
        f"The original code was:\n{old_code}\n\n"
        "Please provide a brief summary of what changes might fix these errors, highlighting key differences."
    )

    try:
        resp = client.chat.completions.create(
            model=ADVANCED_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful code assistant providing concise feedback. Look for whether or not the issue has been considered."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.25
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Refinement summary generation failed: {e}")
        return "No summary available."


def judge_response(user_prompt: str, agent_answer: str) -> dict:
    """
    Evaluate the response using a 'mentor persona' & produce a JSON object with correctness, clarity, best_practices.
    """
    system_msg = {
        "role": "system",
        "content": "You are the Python SDK Mentor. Provide a JSON object with fields correctness, clarity, best_practices."
    }
    user_msg = {
        "role": "user",
        "content": (
            f"User asked:\n{user_prompt}\n\n"
            f"Assistant answered:\n{agent_answer}\n\n"
            "Provide your evaluation as:\n"
            '{ "judge_evaluation": { "result": "..<agent_anser>..", correctness": "...", "clarity": "...", "best_practices": "..." } }'
        )
    }

    try:
        resp = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[system_msg, user_msg],
            temperature=0.2,
            max_tokens=250
        )
        text = resp.choices[0].message.content.strip()
        parsed = json.loads(text)
        store_judge_evaluation(user_prompt, parsed)
        return parsed
    except Exception as e:
        return {"judge_evaluation": {"correctness": "", "clarity": "", "best_practices": f"[Error] {e}"}}


def finalize_task(user_prompt: str, result_summary: str, code_path: str, judgement: dict) -> dict:
    """
    Use GPT-4o-mini to produce a final output JSON with schema:
    {
      "result": <output and/or result from execution of code if available>,
      "summary": <refined summary matching user's style>,
      "code_path": "file://<path/to/script.py>",
      "rating": <scale of 1 to 10 being practical>,
    }
    """
    prompt = (
        f"Based on the following execution details, produce a final output JSON object with the schema. Avoide fallilng in to the trap of providing 'Key Lessons'\n"
        f"{{"
        f"  \"result\": <results of the initial prompt and code execution>,"
        f"  \"summary\": <a well dictated summary that matches the style of the user's original request>,"
        f"  \"code_path\": \"file://{code_path}\","
        f"  \"rating\": <an overall rating based on the judge's evaluation details>"
        f"}}"
        f"Execution details:\nResult Summary: {result_summary}"
        f"Judge Evaluation: {json.dumps(judgement, indent=2)}"
        f"User Request: {user_prompt}\n\nOutput only valid JSON."
    )

    try:
        resp = client.chat.completions.create(
            model=FINAL_MODEL,
            messages=[
                {"role": "system", "content": "You are a final output generator. Create a small summary based on the final output given in the prompt. User the knowledge you have as a smarty pants to give a summary and provide the actual results."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.5
        )
        final_output_text = resp.choices[0].message.content.strip()
        return json.loads(final_output_text)
    except Exception as e:
        logger.warning(f"Final output generation failed: {e}")
        return {
            "result": result_summary,
            "summary": result_summary,
            "code_path": f"file://{code_path}",
            "rating": judgement.get("judge_evaluation", {}).get("correctness", "N/A")
        }

def get_code_prompt(user_prompt: str) -> list[dict[str, str]]:
    # This section sets the context and expectations for the AI model.
    preprompt_padding = """You are a highly skilled Python developer and data scientist. Your responsibility is to generate usable, efficient, and well-structured Python code based on the given requirements. Emphasize clarity, organization, and providing a cohesive solution."""


    # Here is the main task prompt with required format and reinforcement examples.
    preprompt_padding = """You are a highly skilled Python developer. When generating code examples in other languages, use alternative formatting instead of triple backticks when used inside of a code block"""

    # Passed at the end as user instructions along with the user's prompt
    user_instructions = f"""
    Use the following structure to address the task:
    1. **Task Decomposition:** Break down the problem into manageable steps.
    2. **Approach Explanation:** Briefly describe the approach for each step.
    3. **Code Generation:** Provide Python code within triple backticks for each step.
    4. **Always Output:** Results, should always sent to the CLI. Remember you are a CLI tool, no opening displays or cameras or anything like that.

    Example 1:
    - Task: Perform frequency analysis on text.
    - Explanation: Count word occurrences.
    - Code: ```python
    from collections import Counter

    def frequency_analysis(text):
        words = text.split()
        frequency = Counter(words)
        return frequency
    ```    
    Example 2:

    Task: Conduct semantic analysis using NLP techniques.
    Explanation: Use NLP libraries to extract semantic meanings.
    Code: ```
    python import spacy
    nlp = spacy.load('en_core_web_sm') def semantic_analysis(text): doc = nlp(text) for token in doc: print(token.text, token.lemma_, token.pos_, token.dep_)
    ```

    Prompt:
    Given the request "{user_prompt}", execute the following:
    - **Breakdown the task into logical components.**
    - **Implement a solution using Python code.**
    - **Ensure all code snippets are wrapped within triple backticks.**
    - **Ensure all strings and multiline or interpolated strings avoide using backticks.

    Final Note: Always double-check your code for accuracy and completeness.
    """

    # Modify your task_prompt to include this instruction clearly
    task_prompt = f"""
    When showing examples of code in other languages, DO NOT USE TRIPLE BACKTICKS in string literals.
    ALWAYS avoid triple backticks in strings and string literalls. Only use it when showing a whole codeblock:
    

    A NEGATVIE EXAMPLE (DO NOT WRITE CODE LIKE THIS):
    ...
    ...
    ...
    The code below shows how you could store a string of python code with triple ticks
    ```python
    some_code_snippet = "```python
import os
print(os)
```
    ```

    The above example is terrible. Instead when using a string or string interpolaton or anything that results in a string see below:
    WRITE CODE LIKE THIS
    ```python
    some_code_snippet=\"\"\"
import os
print(os)
\"\"\"        
    ```

    This way it is easiest to parse from the response as only executable code is meant to be surrounded by triple backticks.
    No Strings should use triple backticks within any code provided.

    User Instructions and request for code: {user_instructions}


    It is important to remember you are a command line assistant and shouldn't try to display to the screen. Instead use terminal buffering and color enhancing techniques or imports.
    """

    # Combine the components into the messages array.
    logger.info("Classification => code_needed => generating new code with GPT-4.")
    return [
        {"role": "system", "content": preprompt_padding},
        {"role": "user", "content": task_prompt}
    ]
            

def stream_llm_response(model: str, messages: list, temperature=0.35) -> str:
    """
    Stream a ChatCompletion from OpenAI and extract any code in triple backticks.
    """
    try:
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, stream=True)
    except Exception as e:
        logger.warning(f"LLM streaming call failed: {e}")
        return "[LLM Error]"

    full_chunk = ""
    for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_chunk += delta
    print()
    return extract_code_from_chunk(full_chunk)