"""
Shared prompt templates and their detection markers.

Previously the real prompts (in router_agent.py, desk_agent.py,
reflection_agent.py) and mock_backends.py's keyword-sniffing each
hardcoded their own copy of the same marker phrases. Editing a real
prompt could silently break mock-mode detection with no error. Both
sides now import the same constants from this one module.
"""

ROUTER_MARKER = "classify the student"
ANSWER_MARKER = "based on the following context"
REFLECTION_MARKER = "fully supported by the context"

ROUTER_PROMPT = """Classify the student query into exactly one department.
Departments: academic, hostel, examcell, placement, fee.

- academic: syllabus, attendance, courses, credits, timetable, faculty advisors
- hostel: room allotment, mess, hostel fees, warden issues
- examcell: hall tickets, exams, revaluation, arrears, CGPA
- placement: internships, campus drives, resumes, eligibility for placements
- fee: tuition fee payment, scholarships, refunds

Respond with ONLY the single department word — nothing else.

Student query: "{query}"

Department:"""

assert ROUTER_MARKER in ROUTER_PROMPT.lower()

ANSWER_PROMPT = """You are a college helpdesk assistant. Answer the student's
question using ONLY the information in the context below. If the context
does not contain enough information to answer confidently, say so plainly
instead of guessing.

Based on the following context, answer the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

assert ANSWER_MARKER in ANSWER_PROMPT.lower()

REFLECTION_PROMPT = """You are checking whether an answer is fully supported by the context it was
based on. Do not use outside knowledge — judge strictly by whether the
context actually contains what the answer claims.

ANSWER:
{answer}

CONTEXT:
{context}

Respond in exactly this format:
GROUNDED: yes|no
CONFIDENCE: high|low
REASON: <one short sentence>"""

assert REFLECTION_MARKER in REFLECTION_PROMPT.lower()
