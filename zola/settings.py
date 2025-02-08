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
    # Removed the "timeout" label
}

# All standard labels for classification
STANDARD_LABELS = {
    "priority": (
        "For emails needing immediate attention. This may include urgent messages, near-due bills, quick meeting invites, "
        "critical requests, travel bookings needing attention, emails requiring an immediate signature, "
        "or emails from key contacts and familiar senders (`{sender_category}`/`{user_family_name}`)."
    ),
    "respond": (
        "For emails that need a reply or follow-up; can also be 'priority' if urgent. "
        "This includes requests for signatures that aren't time-critical."
    ),
    "review": "For emails to be read or reviewed later; not urgent and not requiring a response.",
    "finance": "For financial matters such as invoices, bills, expenses, and taxes; can be 'priority' if urgent.",
    "travel": "For travel-related emails like bookings, itineraries, and confirmations; use 'priority' if time-sensitive.",
    "school": "For school communications including updates, announcements, and deadlines; mark as 'priority' if urgent.",
    "newsletters": "For regular updates such as newsletters and informational digests.",
    "notifications": (
        "For system-generated notifications such as terms of service updates, privacy policy changes, or service alerts. "
        "These emails are informational and typically do not require immediate action."
    ),
    "completed": (
        "For emails that indicate that a task or conversation is finished. "
        "This may include thank-you notes, confirmations, or other acknowledgements where no further action is needed."
    ),
    "ignore": (
        "For low-priority bulk emails like sales outreach, surveys, promotional content, or automated notifications. "
        "When this label is applied, no other labels should be used."
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
Your task is to analyze each email and apply **exactly one** of these labels:

Available labels:
{LABELS_SECTION}

SLA Details:
{SLA_SECTION}

**Priority and Consistency Guidelines**:
- Consider today's date ({{today_date}}) when assessing urgency.
- Emails from people with your family name ({{user_family_name}}) should be high priority.
- Evaluate typical response times ({{sender_category}}) as well as domain-level patterns ({{domain_category}}).
- Carefully consider if the email seems to be from a human or an automated system. Human messages often warrant 'respond' or 'priority' labels, while automated messages typically go to 'ignore' or 'newsletters' unless they require urgent attention.
- Sales outreach messages should be labeled as 'ignore', even if they appear to be from a human sender.
- Emails that are clearly system-generated notifications (e.g., terms of service updates, privacy policy changes, or service alerts) should be labeled as 'notifications'.
- Emails that signal a completed task or conversation—such as thank-you notes, confirmations, or acknowledgements where no further action is needed—should be labeled as 'completed'.
- The CEO prefers to process similar tasks from the same sender together, so try to categorize all emails from the same sender in a consistent manner.
- For requests involving signatures, mark them as 'respond' if non-urgent, or 'priority' if the signature is time-sensitive.
- If you are uncertain or if multiple labels seem close, choose the single label that makes the most sense.

Email Context (provided solely for analysis):
  From: {{from_header}}
  Subject: {{subject}}
  Received: {{received_age}} ago
  Sender's response pattern: {{sender_category}}
  Domain response pattern: {{domain_category}}

Email Body:
{{email_body}}

**IMPORTANT**:
- Do not include or echo any part of the email context or body in your response.
- Return ONLY a Python list containing exactly one label, for example: ['priority'].
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
