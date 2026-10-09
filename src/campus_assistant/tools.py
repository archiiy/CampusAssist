"""
Custom @tool functions for the follow-up agent.
"""

import re
from datetime import datetime

from langchain_core.tools import tool

from campus_assistant.config import settings
from campus_assistant.rag.retriever import retrieve

DESKS = ["academic", "hostel", "examcell", "placement", "fee"]

MONTH_MAP = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


@tool
def date_calculator(query: str) -> str:
    """Calculate the number of days between today and a given date.
    Pass a natural sentence containing a date, e.g. 'how many days until
    30 September 2026' or 'days left before 15 October 2026'."""
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})", query)
    if not m:
        return "Couldn't find a date in that question. Try a format like '30 September 2026'."
    day, month_name, year = m.groups()
    month = MONTH_MAP.get(month_name.lower())
    if not month:
        return f"Unrecognized month: {month_name}"
    target = datetime(int(year), month, int(day))
    today = datetime.combine(settings.today, datetime.min.time())
    delta = (target - today).days
    if delta > 0:
        return f"{delta} day(s) remaining until {target.strftime('%d %B %Y')}."
    elif delta == 0:
        return f"{target.strftime('%d %B %Y')} is today."
    else:
        return f"{target.strftime('%d %B %Y')} was {-delta} day(s) ago."


@tool
def kb_lookup(desk: str, question: str) -> str:
    """Look up information from a specific department's knowledge base.
    desk must be one of: academic, hostel, examcell, placement, fee."""
    desk = desk.strip().lower()
    if desk not in DESKS:
        return f"Unknown desk '{desk}'. Valid desks: {', '.join(DESKS)}."
    chunks = retrieve(desk, question, top_k=2)
    if not chunks:
        return f"No information found in the {desk} knowledge base for that question."
    return "\n---\n".join(c["text"] for c in chunks)


ALL_TOOLS = [date_calculator, kb_lookup]
