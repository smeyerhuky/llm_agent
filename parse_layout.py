#!/usr/bin/env python3
"""
parse_layout.py

Usage:
  python parse_layout.py project_layout.md

This script scans project_layout.md for headings of the form:

  ## 1. **`some_file.py`**

Then captures ALL triple-backtick code blocks from that heading until
the next heading. We store them in a file under my_llm_agent/some_file.py.
Multiple code blocks per heading are combined in the order they appear.

If you want to preserve snippet tags or subheadings, you can expand the
logic below to search for snippet lines, then embed them as comments or
special markers in the final output.
"""

import sys
import re
import os
from pathlib import Path

# Change this if you want a different output folder:
TARGET_DIR = Path("my_llm_agent")

# Regex for a heading line like:
# ## 1. **`filename.py`** (the part in backticks is group(1))
HEADING_PATTERN = re.compile(r"^##\s+\d+\.\s+\*\*`([^`]+)`\*\*")

# Regex to detect triple-backtick code fences:
# We capture everything inside the backticks (including newlines).
# This handles ```python or just ```:
CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?(.*?)```", re.DOTALL)


def main(md_file: str):
    with open(md_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    TARGET_DIR.mkdir(exist_ok=True, parents=True)

    current_file = None
    section_lines = []

    def flush_section(target_file: str, text: str):
        """
        Given all text lines for a section, find all code blocks and write them
        out to the specified file path, appending them in order.
        """
        code_blocks = CODE_BLOCK_PATTERN.findall(text)
        if not code_blocks:
            print(f"No code blocks found for {target_file}; skipping.")
            return

        # Combine them into one code string, separated by a newline or two.
        combined_code = "\n\n".join(cb.strip("\r\n") for cb in code_blocks)

        out_path = TARGET_DIR / target_file
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as outf:
            outf.write(combined_code)

        print(f"Wrote file: {out_path} with {len(code_blocks)} code block(s).")

    # We'll gather lines for each section. When we encounter a new heading,
    # we flush the old section (if any) to disk, then reset.
    buffer_for_current_heading = []

    for line in lines:
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            # If we're hitting a new heading and there's a previously active file:
            if current_file is not None and buffer_for_current_heading:
                # flush previous
                flush_section(current_file, "".join(buffer_for_current_heading))

            # Start a new section
            current_file = heading_match.group(1).strip()
            buffer_for_current_heading = []
        else:
            # If no heading, just accumulate text for the current section
            if current_file:
                buffer_for_current_heading.append(line)

    # End of file: flush the last section if it exists
    if current_file is not None and buffer_for_current_heading:
        flush_section(current_file, "".join(buffer_for_current_heading))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_layout.py <project_layout.md>")
        sys.exit(1)

    main(sys.argv[1])
