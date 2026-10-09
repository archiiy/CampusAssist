"""
Main entry point — Multi-Agent Campus Assistant and Query Resolution System.

    python main.py               # runs built-in sample queries
    python main.py --mock        # offline demo, no Ollama needed
    python main.py --interactive # type your own query + follow-ups
"""

import argparse
import os
import uuid

from pipeline import build_pipeline_graph
from followup_agent import build_followup_agent, ask
from report import build_report, save_report

SAMPLE_QUERIES = [
    {
        "query": "When is the last date to pay hostel fees this semester?",
        "followups": ["How many days are left until 30 September 2026?"],
    },
    {
        "query": "How many days do I have to apply for revaluation after results are published?",
        "followups": [],
    },
    {
        "query": "Can I bring a pet to the hostel?",
        "followups": [],
    },
]


def run_one_query(pipeline_app, followup_agent, raw_query: str, followups=None, interactive=False, out_dir="reports"):
    session_id = str(uuid.uuid4())[:8]
    result = pipeline_app.invoke({"raw_query": raw_query, "session_id": session_id})

    print(f"\n=== Session {session_id} ===")
    print(f"Query:      {raw_query}")
    print(f"Desk:       {result['desk']}")
    print(f"Grounded:   {result['is_grounded']}  (confidence: {result['confidence']})")
    print(f"Escalated:  {result['escalated']}")
    print(f"Answer:     {result['final_answer']}")

    qna = []
    if interactive:
        print("\nAsk follow-up questions (blank line to finish):")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q:
                break
            a = ask(followup_agent, session_id, q)
            print(a)
            qna.append({"question": q, "answer": a})
    else:
        for q in (followups or []):
            a = ask(followup_agent, session_id, q)
            print(f"  Q: {q}\n  A: {a}")
            qna.append({"question": q, "answer": a})

    report = build_report(result, qna)
    path = save_report(report, out_dir=out_dir)
    print(f"Report saved: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Campus Assistant")
    parser.add_argument("--mock", action="store_true", help="run without Ollama, using the mock LLM")
    parser.add_argument("--interactive", action="store_true", help="type your own query + follow-ups")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    if args.mock:
        os.environ["CAMPUS_ASSISTANT_MOCK"] = "1"

    pipeline_app = build_pipeline_graph()
    followup_agent = build_followup_agent()

    if args.interactive:
        q = input("Ask the campus assistant: ").strip()
        run_one_query(pipeline_app, followup_agent, q, interactive=True, out_dir=args.out)
    else:
        for item in SAMPLE_QUERIES:
            run_one_query(pipeline_app, followup_agent, item["query"], followups=item["followups"], out_dir=args.out)


if __name__ == "__main__":
    main()
