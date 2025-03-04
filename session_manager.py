# session_manager.py

import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Holds session-wide state such as:
    - A dictionary of user prompts and previous responses.
    - Possibly other session-related info.
    """

    def __init__(self):
        # This dictionary can store arbitrary session context:
        # e.g. { "prompt1": response_data, "module:modname": imported_module, ... }
        self.context = {}

    def set(self, key: str, value):
        self.context[key] = value

    def get(self, key: str):
        return self.context.get(key)

    def items(self):
        return self.context.items()

    def __contains__(self, key: str):
        return key in self.context

    def __getitem__(self, key):
        return self.context[key]

    def __setitem__(self, key, value):
        self.context[key] = value

    def __delitem__(self, key):
        del self.context[key]

    def dump_context(self):
        """
        Return a string representation of all stored context, for debugging or logging.
        """
        lines = []
        for k, v in self.context.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)