"""
Builds the checkpointer used for follow-up conversation memory.

"sqlite" (the default) persists sessions to a local file so a restart
of the Streamlit app or CLI doesn't silently lose follow-up context.
"memory" (non-persistent, in-process only) is used by the test suite
so pytest runs don't leave .sqlite files behind.
"""

import sqlite3

from langgraph.checkpoint.memory import MemorySaver

from campus_assistant.config import settings


def get_checkpointer():
    if settings.checkpointer_backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        # SqliteSaver.from_conn_string(...) is a context manager that closes
        # the connection on exit — unsuitable here since this checkpointer
        # must outlive the call (it's reused across the app/CLI process), so
        # the connection is opened directly and left open for the process.
        conn = sqlite3.connect(str(settings.checkpointer_sqlite_path), check_same_thread=False)
        return SqliteSaver(conn)
    return MemorySaver()
