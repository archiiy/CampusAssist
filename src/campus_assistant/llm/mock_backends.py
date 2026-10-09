"""
Mock chat model for offline testing (mock=True / CAMPUS_ASSISTANT_MOCK=1).

Handles four prompt "modes" by sniffing marker phrases shared with the
real prompts in campus_assistant.prompts (so editing a real prompt and
breaking mock detection can't happen silently — see prompts.py):
  - classification   -> returns a desk label
  - RAG answer draft  -> synthesizes a short answer from provided context
  - reflection check  -> decides grounded/ungrounded + confidence
  - follow-up chat    -> ReAct-style tool calling

This is intentionally simple pattern-matching, not real understanding —
use the real qwen2.5:3b model to actually judge quality.
"""

import re
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from campus_assistant.prompts import ROUTER_MARKER, ANSWER_MARKER, REFLECTION_MARKER

DESK_KEYWORDS = {
    "hostel": ["hostel", "room", "mess", "warden", "roommate"],
    "examcell": ["exam", "hall ticket", "revaluation", "arrear", "cgpa"],
    "placement": ["placement", "internship", "recruiter", "resume", "campus drive", "package"],
    "fee": ["fee", "payment", "due", "scholarship", "refund", "tuition"],
    "academic": ["syllabus", "attendance", "course", "credit", "faculty", "timetable"],
}


class MockChatModel(Runnable):
    def __init__(self):
        self._tools = {}

    def bind_tools(self, tools, **kwargs):
        self._tools = {t.name: t for t in tools}
        return self

    def invoke(self, prompt, config=None, **kwargs):
        if isinstance(prompt, list):
            return self._invoke_chat(prompt)

        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        low = text.lower()

        if ROUTER_MARKER in low:
            return self._classify(low)

        if ANSWER_MARKER in low:
            return self._draft_answer(text)

        if REFLECTION_MARKER in low:
            return self._reflect(text)

        return AIMessage("I'm not sure — this is the mock model, swap in a real one.")

    def _classify(self, low_prompt: str) -> AIMessage:
        m = re.search(r'student query:\s*"(.*?)"', low_prompt, re.DOTALL)
        query = m.group(1) if m else low_prompt
        scores = {}
        for desk, keywords in DESK_KEYWORDS.items():
            # weight by keyword length so a specific phrase match (e.g.
            # "placement drive") outweighs a short, more ambiguous one
            # (e.g. "cgpa" appearing incidentally in a different desk's list)
            scores[desk] = sum(len(k) for k in keywords if k in query)
        best_desk = max(scores, key=scores.get)
        if scores[best_desk] == 0:
            return AIMessage("academic")
        return AIMessage(best_desk)

    def _draft_answer(self, text: str) -> AIMessage:
        m = re.search(r"CONTEXT:\s*(.*?)\n\nQUESTION:", text, re.DOTALL)
        context = m.group(1).strip() if m else ""
        qm = re.search(r"QUESTION:\s*(.*)", text, re.DOTALL)
        question = qm.group(1).strip() if qm else ""

        if not context:
            return AIMessage("I don't have enough information in the available records to answer this confidently.")

        question_words = set(re.findall(r"[a-z]{4,}", question.lower()))
        # split into sentence-ish chunks rather than raw lines, so we can
        # score relevance against the actual question instead of just
        # grabbing whatever text happens to be first (e.g. a doc header)
        candidates = re.split(r"(?<=[.:])\s+|\n+", context)
        candidates = [c.strip() for c in candidates if len(c.strip()) > 15]

        best_sentence, best_score = None, 0
        for sentence in candidates:
            sentence_words = set(re.findall(r"[a-z]{4,}", sentence.lower()))
            score = len(question_words & sentence_words)
            if score > best_score:
                best_sentence, best_score = sentence, score

        if not best_sentence or best_score < 2:
            return AIMessage("I don't have enough information in the available records to answer this confidently.")
        return AIMessage(f"Based on the available records: {best_sentence}")

    def _reflect(self, text: str) -> AIMessage:
        m = re.search(r"ANSWER:\s*(.*?)\n\nCONTEXT:", text, re.DOTALL)
        answer = (m.group(1) if m else "").lower()
        cm = re.search(r"CONTEXT:\s*(.*)", text, re.DOTALL)
        context = (cm.group(1) if cm else "").lower()
        if "don't have enough information" in answer or not context.strip():
            return AIMessage("GROUNDED: no\nCONFIDENCE: low\nREASON: no supporting context retrieved")
        # crude grounding check: does the answer share enough words with the context?
        answer_words = set(re.findall(r"[a-z]{4,}", answer))
        context_words = set(re.findall(r"[a-z]{4,}", context))
        overlap = len(answer_words & context_words)
        if overlap >= 3:
            return AIMessage("GROUNDED: yes\nCONFIDENCE: high\nREASON: answer closely matches retrieved context")
        return AIMessage("GROUNDED: no\nCONFIDENCE: low\nREASON: answer does not clearly match retrieved context")

    def _invoke_chat(self, messages):
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return AIMessage(content=f"Here's what I found: {last.content}")

        question = last.content.lower() if isinstance(last, HumanMessage) else ""

        if "date_calculator" in self._tools and any(w in question for w in ["days", "how long", "deadline", "when", "left until"]):
            return AIMessage(content="", tool_calls=[{
                "name": "date_calculator", "args": {"query": last.content}, "id": "call_1",
            }])

        if "kb_lookup" in self._tools:
            desk = next((d for d in DESK_KEYWORDS if any(k in question for k in DESK_KEYWORDS[d])), "academic")
            return AIMessage(content="", tool_calls=[{
                "name": "kb_lookup", "args": {"desk": desk, "question": last.content}, "id": "call_1",
            }])

        return AIMessage(content="I'm not sure — this is the mock model, swap in a real one.")
