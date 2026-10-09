"""
Desk Agent (RAG).

One generic agent function parameterized by desk name — retrieves the
top matching chunks from that desk's vector store, then asks the LLM
to answer USING ONLY that retrieved context.
"""

from campus_assistant.state import CampusAssistState
from campus_assistant.llm.factory import get_llm
from campus_assistant.rag.retriever import retrieve
from campus_assistant.prompts import ANSWER_PROMPT


def desk_agent(state: CampusAssistState, mock: bool | None = None) -> CampusAssistState:
    desk = state["desk"]
    query = state["raw_query"]

    chunks = retrieve(desk, query, top_k=3)
    context = "\n\n".join(c["text"] for c in chunks) if chunks else ""

    llm = get_llm(mock=mock)
    prompt = ANSWER_PROMPT.format(context=context, question=query)
    raw = llm.invoke(prompt)
    answer = raw.content if hasattr(raw, "content") else str(raw)

    return {
        "retrieved_chunks": chunks,
        "draft_answer": answer.strip(),
    }


if __name__ == "__main__":
    test_cases = [
        ("hostel", "When is the last date to pay hostel fees?"),
        ("examcell", "How many days do I have to apply for revaluation?"),
        ("placement", "What is the minimum CGPA needed for placements?"),
        ("hostel", "Can I bring a pet to the hostel?"),  # not in the docs — should say so
    ]
    for desk, query in test_cases:
        result = desk_agent({"desk": desk, "raw_query": query})
        print(f"[{desk}] {query}")
        print(f"  -> {result['draft_answer']}")
        print(f"  (retrieved {len(result['retrieved_chunks'])} chunks)\n")
