"""Seed data: 3 fictional companies x 4 users x 5 documents.

Every tenant has a salary sheet on purpose: "What are the salaries?" is then a
semantically perfect match for ALL tenants, which is exactly the leak we guard against.
Each document's first line carries a unique CANARY marker; if a canary ever shows up
in an answer or prompt for someone not allowed to see that document, the test fails.
"""
from .models import Document, User

ALL_STAFF = ("employee", "manager", "hr", "admin")
STAFF_INTERNAL = ("employee", "manager", "hr", "admin")

_TENANTS = {
    # tenant: (company, what they do, salary L1, salary L3, restructuring target)
    "acme": ("Acme Robotics", "industrial warehouse robots", "62,000", "118,000",
             "merge the Pune and Austin assembly teams"),
    "globex": ("Globex Bank", "retail banking and loans", "58,000", "131,000",
               "close eleven low-traffic branches"),
    "initech": ("Initech Software", "payroll software for mid-size firms", "71,000", "142,000",
                "sunset the legacy desktop product line"),
}


def _docs_for(t: str) -> list[Document]:
    company, domain, l1, l3, restruct = _TENANTS[t]
    emp = f"{t}-employee"
    tag = t.upper()
    return [
        Document(f"{t}-public-overview", t, f"{company} company overview", "public",
                 ALL_STAFF, (),
                 f"{company} company overview (ref CANARY-{tag}-PUBLIC)\n\n"
                 f"{company} is a company focused on {domain}. We publish our press "
                 f"releases and product news on our website.\n\n"
                 f"Our mission is to make {domain} simpler and safer for customers."),
        Document(f"{t}-internal-handbook", t, f"{company} employee handbook", "internal",
                 STAFF_INTERNAL, (),
                 f"{company} employee handbook (ref CANARY-{tag}-INTERNAL)\n\n"
                 f"Leave policy: employees get 24 days of paid leave per year. "
                 f"Pay reviews happen every March and bonuses are paid in April.\n\n"
                 f"Expense policy: submit travel and equipment expenses within 30 days "
                 f"through the finance portal. Managers approve claims under 500 dollars."),
        Document(f"{t}-hr-salary-sheet", t, f"{company} HR salary sheet", "confidential",
                 ("hr",), (),
                 f"{company} HR salary sheet (ref CANARY-{tag}-SALARY)\n\n"
                 f"Confidential salaries by level. Level L1 engineer salary: {l1} USD. "
                 f"Level L3 senior engineer salary: {l3} USD.\n\n"
                 f"Individual salary adjustments for this year are listed per employee "
                 f"in the compensation appendix. Do not share outside HR."),
        Document(f"{t}-mgr-restructuring", t, f"{company} restructuring plan", "confidential",
                 ("manager", "hr"), (),
                 f"{company} restructuring plan (ref CANARY-{tag}-RESTRUCTURE)\n\n"
                 f"Confidential plan for next quarter: {restruct}. Managers must not "
                 f"discuss the plan with their teams until HR announces it.\n\n"
                 f"Expected headcount impact and severance budget are listed in appendix B."),
        Document(f"{t}-review-{emp}", t, f"Performance review for {emp}", "confidential",
                 ("hr",), (),          # HR-only: employees can not read their own review
                 f"Performance review for {emp} (ref CANARY-{tag}-REVIEW)\n\n"
                 f"Annual performance review. Rating: exceeds expectations. "
                 f"Strengths: ownership, clear communication. Growth area: delegation.\n\n"
                 f"Recommended promotion to the next level and a salary increase."),
    ]


TENANT_IDS = tuple(_TENANTS)
DOCUMENTS: list[Document] = [d for t in TENANT_IDS for d in _docs_for(t)]

USERS: dict[str, User] = {}
for _t in TENANT_IDS:
    USERS[f"{_t}-employee"] = User(f"{_t}-employee", _t, ("employee",))
    USERS[f"{_t}-manager"] = User(f"{_t}-manager", _t, ("employee", "manager"))
    USERS[f"{_t}-hr"] = User(f"{_t}-hr", _t, ("employee", "hr"))
    USERS[f"{_t}-admin"] = User(f"{_t}-admin", _t, ("employee", "admin"))