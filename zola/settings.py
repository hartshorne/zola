SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.modify",
]

# Parent labels used for organizing Gmail labels
PARENT_LABEL = "zola"
LLM_PARENT_LABEL = f"{PARENT_LABEL}/🤖"

# LLM configuration
MAX_OUTPUT_TOKENS = 1024
MAX_INPUT_TOKENS = 8192

# Error handling labels - used for system-level error tracking
ERROR_LABELS = {
    f"{LLM_PARENT_LABEL}/error": "When LLM processing fails with an exception",
    f"{LLM_PARENT_LABEL}/confused": "When LLM cannot confidently apply standard labels",
    f"{LLM_PARENT_LABEL}/timeout": "When LLM processing times out",
}

# All standard labels for classification
STANDARD_LABELS = {
    "priority": (
        "For items requiring immediate attention, such as:\n"
        "  * Messages from people sharing your family name (`{user_family_name}`)\n"
        "  * Invoices/bills due within 5 days\n"
        "  * Meeting invites for today or tomorrow\n"
        "  * Urgent requests from key contacts\n"
        "  * Travel bookings needing immediate action\n"
        "  * Messages from senders you typically respond to quickly (`{sender_category}`)"
    ),
    "respond": ("For emails that need a reply or follow-up.\nCan be combined with 'priority' if urgent."),
    "review": ("For items to read or review when time permits.\nNot immediately urgent or requiring a response."),
    "finance": (
        "For invoices, bills, expenses, and other financial matters.\n"
        "Can be combined with 'priority' if due soon or urgent."
    ),
    "travel": (
        "For travel-related emails (bookings, itineraries, confirmations).\n"
        "Can be combined with 'priority' if departure is imminent."
    ),
}

# All valid labels that can be applied to messages
ALL_LABELS = {
    # Parent labels
    PARENT_LABEL: "Parent label for all Zola labels",
    LLM_PARENT_LABEL: "Parent label for LLM-related labels",
    # Standard labels
    **{f"{PARENT_LABEL}/{name}": desc for name, desc in STANDARD_LABELS.items()},
    # Error labels
    **ERROR_LABELS,
}


def generate_labels_markdown(labels_dict: dict) -> str:
    """
    Dynamically generates the markdown text for each label and its description.
    """
    lines = []
    for label_name, label_description in labels_dict.items():
        lines.append(f"- **{label_name}**:\n{label_description}\n")
    return "\n".join(lines).strip()


# This variable contains the expanded markdown for each label:
LABELS_SECTION = generate_labels_markdown(STANDARD_LABELS)

# Define SLA categories in ascending order of response time, including a short description.
# The structure is: (threshold_in_seconds, category_name, short_description)
SLA_CATEGORIES = [
    (1800, "immediate", "within 30 minutes"),
    (7200, "quick", "within 2 hours"),
    (28800, "same_day", "within 8 hours"),
    (86400, "next_day", "within 24 hours"),
    (float("inf"), "delayed", "longer than 24 hours"),
]


def generate_sla_markdown(sla_categories) -> str:
    """
    Dynamically generates a brief explanation of the SLA categories
    without repeating data in multiple places.
    """
    lines = [
        "We use heuristics to track how quickly the CEO typically responds to a given sender.",
        "Here are the categories (SLA) in ascending order of typical response time:",
    ]
    for _, category_name, category_description in sla_categories:
        lines.append(f"- **{category_name}**: {category_description}")
    lines.append(
        "\nYou may see something like `Response Pattern for sender: <SLA_CATEGORY>` "
        "indicating how quickly emails from that sender are normally answered. "
        "Use this to help gauge urgency (e.g., 'immediate' or 'quick' often suggests higher priority)."
    )
    return "\n".join(lines)


# Generate the SLA section once, referencing the single SLA_CATEGORIES source of truth
SLA_SECTION = generate_sla_markdown(SLA_CATEGORIES)

# LLM prompt template that uses the labels markdown and SLA explanation
LLM_CONTEXT_TEMPLATE = f"""You are an elite executive assistant helping to organize an inbox using a simple but effective labeling system.
Your task is to analyze each email and apply one or more of these labels:

Available labels (can apply multiple):
{LABELS_SECTION}

{SLA_SECTION}

Priority Guidelines:
- Consider today's date ({{today_date}}) when deciding urgency
- Messages from people with your family name ({{user_family_name}}) should be priority
- Consider the sender's typical response time ({{sender_category}})
- Evaluate domain-level patterns ({{domain_category}})

Return ONLY a Python list of labels, e.g.: ['priority', 'finance']

Email Context:
  From: {{from_header}}
  Subject: {{subject}}
  Received: {{received_age}} ago
  Response Pattern for sender: {{sender_category}}
  Response Pattern for {{sender_domain}}: {{domain_category}}

Email Body:
{{email_body}}
"""

# Default SLA in seconds (1 hour)
DEFAULT_SLA_SECONDS = 3600


def _get_sla_category(self, sla_value: float, has_data: bool) -> str:
    """
    Convert an SLA value (in seconds) to a category name, based on SLA_CATEGORIES.
    If 'has_data' is False, returns 'never_responded'.
    """
    if not has_data:
        return "never_responded"
    for threshold, category_name, _ in SLA_CATEGORIES:
        if sla_value <= threshold:
            return category_name
    return "delayed"
