from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiofiles
from aiogoogle.client import Aiogoogle
from humanize import naturaldelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from zola.account_manager import UserInfo
from zola.gmail_utils import (
    extract_email_domain,
    format_datetime_for_model,
    get_email_body_markdown,
    get_message,
    get_or_create_label,
    get_user_timezone,
    list_messages,
    parse_date,
    update_message_labels,
)
from zola.llm_utils import classify_email
from zola.settings import (
    DEFAULT_SLA_SECONDS,
    LLM_CONTEXT_TEMPLATE,
    LLM_PARENT_LABEL,
    PARENT_LABEL,
    SLA_CATEGORIES,
    STANDARD_LABELS,
)

logger = logging.getLogger(__name__)
console = Console()


@dataclass(frozen=True)
class TimeBucket:
    name: str
    duration: timedelta
    weight: float


# Define our buckets as a constant list
BUCKETS: List[TimeBucket] = [
    TimeBucket("hour", timedelta(hours=1), 1.0),
    TimeBucket("day", timedelta(days=1), 0.7),
    TimeBucket("week", timedelta(days=7), 0.5),
    TimeBucket("month", timedelta(days=30), 0.3),
]


def build_sla_table(
    data: Dict[str, Dict[str, List[float]]],
    known: Set[str],
    buckets: List[TimeBucket],
    title: str,
    sla_data: SLAData,
) -> Table:
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column(title.split()[0])
    for bucket in buckets:
        table.add_column(f"{bucket.name}\n(w={bucket.weight:.1f})")
    table.add_column("Overall SLA")
    table.add_column("Category")

    for key in sorted(known):
        row = [key]
        buckets_data = data.get(key, {})
        if buckets_data:
            total_responses = sum(len(times) for times in buckets_data.values())
            for bucket in buckets:
                times = buckets_data.get(bucket.name, [])
                if times:
                    count = len(times)
                    avg = sum(times) / count
                    resp_weight = count / total_responses
                    row.append(f"{count} resp\n{naturaldelta(avg)}\n(w={bucket.weight * resp_weight:.2f})")
                else:
                    row.append("-")
            overall_sla = sla_data._calculate_weighted_sla(buckets_data)
            row.append(naturaldelta(overall_sla) if overall_sla != float("inf") else "no data")
            row.append(sla_data._get_sla_category(overall_sla, True))
        else:
            row.extend(["-"] * (len(buckets) + 2))
        table.add_row(*row)
    return table


class SLAData:
    def __init__(self) -> None:
        # sender_data and domain_data are keyed by sender/domain and then bucket name
        self.sender_data: Dict[str, Dict[str, List[float]]] = {}
        self.domain_data: Dict[str, Dict[str, List[float]]] = {}
        self.known_senders: Set[str] = set()
        self.known_domains: Set[str] = set()

    def _get_sla_category(self, sla_value: float, has_data: bool) -> str:
        if not has_data:
            return "never_responded"
        for threshold, category_name, _ in SLA_CATEGORIES:
            if sla_value <= threshold:
                return category_name
        return "delayed"

    def add_response_time(self, sender: str, domain: str, response_time: float, timestamp: datetime) -> None:
        now = datetime.now(timezone.utc)
        age = now - timestamp
        self.known_senders.add(sender)
        self.known_domains.add(domain)
        if sender not in self.sender_data:
            self.sender_data[sender] = {bucket.name: [] for bucket in BUCKETS}
        if domain not in self.domain_data:
            self.domain_data[domain] = {bucket.name: [] for bucket in BUCKETS}
        for bucket in BUCKETS:
            if age <= bucket.duration:
                self.sender_data[sender][bucket.name].append(response_time)
                self.domain_data[domain][bucket.name].append(response_time)
        bucket_names = [bucket.name for bucket in BUCKETS if age <= bucket.duration]
        logger.debug(
            f"Added response time:\n  Sender: {sender}\n  Domain: {domain}\n"
            f"  Response Time: {naturaldelta(response_time)}\n  Age: {naturaldelta(age)}\n"
            f"  Buckets: {bucket_names}"
        )

    def _calculate_weighted_sla(self, bucket_data: Dict[str, List[float]]) -> float:
        total_responses = sum(len(bucket_data.get(bucket.name, [])) for bucket in BUCKETS)
        if total_responses == 0:
            return float("inf")
        weighted_sum = 0.0
        total_weight = 0.0
        for bucket in BUCKETS:
            times = bucket_data.get(bucket.name, [])
            if times:
                count = len(times)
                avg = sum(times) / count
                combined_weight = bucket.weight * (count / total_responses)
                weighted_sum += avg * combined_weight
                total_weight += combined_weight
        return weighted_sum / total_weight if total_weight > 0 else float("inf")

    def get_weighted_sla(self, sender: str, domain: str) -> Tuple[float, float, bool, bool]:
        has_sender_data = sender in self.sender_data
        has_domain_data = domain in self.domain_data
        sender_sla = self._calculate_weighted_sla(self.sender_data.get(sender, {}))
        domain_sla = self._calculate_weighted_sla(self.domain_data.get(domain, {}))
        return sender_sla, domain_sla, has_sender_data, has_domain_data

    def get_sla_categories(self, sender: str, domain: str) -> Tuple[str, str]:
        sender_sla, domain_sla, has_sender_data, has_domain_data = self.get_weighted_sla(sender, domain)
        return (
            self._get_sla_category(sender_sla, has_sender_data),
            self._get_sla_category(domain_sla, has_domain_data),
        )

    def print_debug_stats(self) -> None:
        logger.info("\nSLA Statistics Summary:")
        sender_table = build_sla_table(self.sender_data, self.known_senders, BUCKETS, "Sender Response Times", self)
        domain_table = build_sla_table(self.domain_data, self.known_domains, BUCKETS, "Domain Response Times", self)
        console.print("\n")
        console.print(sender_table)
        console.print("\n")
        console.print(domain_table)
        console.print("\n")


async def get_user_info(aiogoogle: Aiogoogle) -> UserInfo:
    if hasattr(aiogoogle, "_user_info"):
        logger.info("User info found in aiogoogle cache.")
        return aiogoogle._user_info  # type: ignore

    accounts_path = Path("secrets/accounts.json")
    if accounts_path.exists():
        logger.info("Loading accounts data from %s", accounts_path)
        async with aiofiles.open(accounts_path, "r") as f:
            content = await f.read()
            accounts = json.loads(content)
    else:
        logger.info("No %s found; using empty data.", accounts_path)
        accounts = {}

    gmail = await aiogoogle.discover("gmail", "v1")
    profile = await aiogoogle.as_user(gmail.users.getProfile(userId="me"))
    my_email = profile.get("emailAddress", "me")
    logger.info("Detected email from Gmail profile: %s", my_email)

    for account_name, account_info in accounts.items():
        if account_info.get("email") == my_email:
            user_name = account_info.get("name", "User")
            user_info = UserInfo(shortname=account_name, name=user_name, email=my_email)
            logger.info(
                "Matched user info from accounts.json -> shortname=%s, name=%s, email=%s",
                account_name,
                user_name,
                my_email,
            )
            aiogoogle._user_info = user_info
            return user_info

    logger.info("No account found for %s in accounts.json. Prompting user...", my_email)
    user_name = input("Enter your name: ").strip()
    account_name = input("Enter a short name for this account (e.g. 'personal' or 'work'): ").strip().lower()

    while account_name in accounts:
        account_name = (
            input(f"Account name '{account_name}' already exists. Please choose another name: ").strip().lower()
        )

    accounts[account_name] = {"email": my_email, "name": user_name}
    async with aiofiles.open(accounts_path, "w") as f:
        await f.write(json.dumps(accounts, indent=2))
    logger.info(
        "Saved new user information to accounts.json -> shortname=%s, name=%s, email=%s",
        account_name,
        user_name,
        my_email,
    )
    user_info = UserInfo(shortname=account_name, name=user_name, email=my_email)
    aiogoogle._user_info = user_info
    return user_info


async def process_sent_message(
    aiogoogle: Aiogoogle,
    msg_meta: Dict[str, Any],
    pbar: tqdm,
    sla_data: SLAData,
    my_email: str,
) -> None:
    """Process a sent message to calculate response times."""
    msg_id = msg_meta["id"]
    logger.info("Processing 'sent' message with ID: %s", msg_id)
    try:
        message = await get_sent_message_metadata(aiogoogle, msg_id)
        headers = message.get("payload", {}).get("headers", [])

        sent_date = await get_sent_date(headers, msg_id)
        if not sent_date:
            return

        in_reply_to = next((h["value"] for h in headers if h["name"].lower() == "in-reply-to"), None)
        if in_reply_to:
            await process_reply_message(aiogoogle, message, headers, sent_date, sla_data, my_email)
        else:
            process_new_message(headers, sent_date, sla_data)

    except Exception as e:
        logger.warning("Error processing sent message %s: %s", msg_id, str(e))
    finally:
        pbar.update(1)


async def get_sent_message_metadata(aiogoogle: Aiogoogle, msg_id: str) -> Dict[str, Any]:
    """Get metadata for a sent message."""
    gmail = await aiogoogle.discover("gmail", "v1")
    return await aiogoogle.as_user(
        gmail.users.messages.get(
            userId="me",
            id=msg_id,
            format="metadata",
            metadataHeaders=["date", "in-reply-to", "to", "from"],
        )
    )


async def get_sent_date(headers: List[Dict[str, str]], msg_id: str) -> Optional[datetime]:
    """Extract and parse the sent date from message headers."""
    sent_date_str = next((h["value"] for h in headers if h["name"].lower() == "date"), None)
    if not sent_date_str:
        logger.info("Message %s does not contain a Date header; skipping SLA calculation.", msg_id)
        return None

    sent_date = parse_date(sent_date_str)
    logger.info("Message %s sent at %s", msg_id, sent_date_str)
    return sent_date


async def process_reply_message(
    aiogoogle: Aiogoogle,
    message: Dict[str, Any],
    headers: List[Dict[str, str]],
    sent_date: datetime,
    sla_data: SLAData,
    my_email: str,
) -> None:
    """Process a message that is a reply to calculate response time."""
    logger.info("Found 'In-Reply-To' header -> searching original message in same thread.")
    thread_msgs = await get_thread_messages(aiogoogle, message["threadId"])

    candidates = get_candidate_messages(thread_msgs, sent_date, my_email)
    if not candidates:
        logger.info("No valid original message found for computing reply-time.")
        return

    candidate_date, candidate = max(candidates, key=lambda x: x[0])
    candidate_headers = candidate.get("payload", {}).get("headers", [])
    from_header = next((h["value"] for h in candidate_headers if h["name"].lower() == "from"), "")

    update_sla_data(sla_data, from_header, sent_date, candidate_date)


async def get_thread_messages(aiogoogle: Aiogoogle, thread_id: str) -> List[Dict[str, Any]]:
    """Get all messages in a thread."""
    gmail = await aiogoogle.discover("gmail", "v1")
    thread = await aiogoogle.as_user(
        gmail.users.threads.get(
            userId="me",
            id=thread_id,
            format="metadata",
            metadataHeaders=["date", "from"],
        )
    )
    thread_msgs = thread.get("messages", [])
    logger.info("Retrieved %d messages in thread %s", len(thread_msgs), thread_id)
    return thread_msgs


def get_candidate_messages(
    thread_msgs: List[Dict[str, Any]],
    sent_date: datetime,
    my_email: str,
) -> List[Tuple[datetime, Dict[str, Any]]]:
    """Get candidate messages that could be the original message being replied to."""
    candidates = []
    for msg in thread_msgs:
        msg_headers = msg.get("payload", {}).get("headers", [])
        from_header = next((h["value"] for h in msg_headers if h["name"].lower() == "from"), "")
        if my_email.lower() in from_header.lower():
            continue

        msg_date_str = next((h["value"] for h in msg_headers if h["name"].lower() == "date"), None)
        if not msg_date_str:
            continue

        msg_date = parse_date(msg_date_str)
        if msg_date < sent_date:
            candidates.append((msg_date, msg))

    return candidates


def process_new_message(
    headers: List[Dict[str, str]],
    sent_date: datetime,
    sla_data: SLAData,
) -> None:
    """Process a new message (not a reply)."""
    to_header = next((h["value"] for h in headers if h["name"].lower() == "to"), "")
    logger.info("Message starts a new conversation; using default SLA=%s.", DEFAULT_SLA_SECONDS)
    update_sla_data(sla_data, to_header, sent_date, None)


def update_sla_data(
    sla_data: SLAData,
    header: str,
    sent_date: datetime,
    received_date: Optional[datetime] = None,
) -> None:
    """Update SLA data with timing information."""
    sender = header
    domain = extract_email_domain(header)

    if received_date:
        reply_time = (sent_date - received_date).total_seconds()
        logger.info(
            "Processed message: reply time %s seconds for sender [%s] (domain=%s).",
            reply_time,
            sender,
            domain,
        )
    else:
        reply_time = DEFAULT_SLA_SECONDS

    sla_data.add_response_time(sender, domain, reply_time, sent_date)


async def process_sent_folder(
    aiogoogle: Aiogoogle,
    sla_data: SLAData,
    max_messages: int,
) -> None:
    logger.info("Retrieving up to %d messages from [SENT] label...", max_messages)
    messages = await list_messages(aiogoogle, "SENT", max_messages)
    logger.info("Found %d sent messages to process.", len(messages))
    user_info = await get_user_info(aiogoogle)
    with tqdm(total=len(messages), desc="Sent messages", unit="msg") as pbar:
        tasks = [
            asyncio.create_task(process_sent_message(aiogoogle, msg_meta, pbar, sla_data, user_info.email))
            for msg_meta in messages
        ]
        if tasks:
            logger.info("Beginning async gather for %d tasks in [SENT].", len(tasks))
            await asyncio.gather(*tasks)
    logger.info("Finished processing sent messages.")


def normalize_label(label: str) -> str:
    lower_label = label.lower()
    if lower_label in STANDARD_LABELS:
        return lower_label
    return lower_label.replace("_", " ")


def apply_parent_label(label: str) -> str:
    """Ensure that the label starts with the correct parent prefix."""
    base = PARENT_LABEL
    if not label.lower().startswith(f"{base.lower()}/"):
        label = f"{base}/{label}"
    double_prefix = f"{base}/{base}/"
    if label.startswith(double_prefix):
        label = label.replace(double_prefix, f"{base}/", 1)
    return label


async def process_inbox_message(
    aiogoogle: Aiogoogle,
    msg_id: str,
    sla_data: SLAData,
    model: Any,
    tokenizer: Any,
) -> List[str]:
    """Process an inbox message and return list of labels to apply."""
    try:
        message = await _fetch_message_data(aiogoogle, msg_id)
        if message is None:
            return [f"{LLM_PARENT_LABEL}/error"]

        headers = _extract_headers(message)
        logger.info(
            "Inbox message %s subject=[%s], from=[%s], to=[%s], date=[%s]",
            msg_id,
            headers["subject"],
            headers["from"],
            headers["to"],
            headers["date"],
        )

        age_info = _get_message_age(headers["date"])
        logger.info(
            "Inbox message %s is about %s old (parsed date=%s).", msg_id, age_info["age_str"], age_info["datetime"]
        )

        sender_info = _get_sender_info(headers["from"], sla_data)
        logger.info(
            "For message %s, SLA categories => sender=%s / domain=%s.",
            msg_id,
            sender_info["sender_category"],
            sender_info["domain_category"],
        )

        context = await _build_message_context(aiogoogle, message, headers, age_info, sender_info, tokenizer)
        logger.info(
            "Message %s final context has %d tokens (~%.1f%% of model limit=%d).",
            msg_id,
            context["token_count"],
            (context["token_count"] / tokenizer.model_max_length) * 100,
            tokenizer.model_max_length,
        )

        return await _get_final_labels(msg_id, context["text"], model, tokenizer)

    except Exception as e:
        logger.error("Error processing message %s: %s", msg_id, str(e), exc_info=True)
        return [f"{LLM_PARENT_LABEL}/error"]


async def _fetch_message_data(aiogoogle: Aiogoogle, msg_id: str) -> Optional[Dict[str, Any]]:
    """Fetch full message data from Gmail API."""
    logger.info("Fetching full data for inbox message %s ...", msg_id)
    return await get_message(
        aiogoogle,
        msg_id,
        format="full",
        metadata_headers=["From", "To", "Subject", "Date"],
    )


def _extract_headers(message: Dict[str, Any]) -> Dict[str, str]:
    """Extract relevant headers from message."""
    headers = message.get("payload", {}).get("headers", [])
    return {
        "subject": next((h["value"] for h in headers if h["name"].lower() == "subject"), ""),
        "to": next((h["value"] for h in headers if h["name"].lower() == "to"), ""),
        "from": next((h["value"] for h in headers if h["name"].lower() == "from"), ""),
        "date": next((h["value"] for h in headers if h["name"].lower() == "date"), ""),
    }


def _get_message_age(date_header: str) -> Dict[str, Any]:
    """Calculate message age information."""
    now = datetime.now(timezone.utc)
    if date_header:
        email_datetime = parse_date(date_header)
        time_diff = now - email_datetime
        age_str = naturaldelta(time_diff)
    else:
        email_datetime = now
        age_str = "unknown age"

    return {"datetime": email_datetime, "age_str": age_str, "now": now}


def _get_sender_info(from_header: str, sla_data: SLAData) -> Dict[str, str]:
    """Get sender and domain categorization."""
    sender_domain = extract_email_domain(from_header)
    sender_category, domain_category = sla_data.get_sla_categories(from_header, sender_domain)
    return {
        "sender": from_header,
        "domain": sender_domain,
        "sender_category": sender_category,
        "domain_category": domain_category,
    }


async def _build_message_context(
    aiogoogle: Aiogoogle,
    message: Dict[str, Any],
    headers: Dict[str, str],
    age_info: Dict[str, Any],
    sender_info: Dict[str, str],
    tokenizer: Any,
) -> Dict[str, Any]:
    """Build context for LLM classification."""
    user_info = await get_user_info(aiogoogle)
    email_body_md = await get_email_body_markdown(aiogoogle, message)

    context_template = _get_base_context(user_info, headers, age_info, sender_info)

    email_body = _truncate_email_body(email_body_md, context_template, tokenizer)
    final_context = context_template.format(email_body=email_body)

    return {"text": final_context, "token_count": len(tokenizer.encode(final_context))}


def _get_base_context(
    user_info: Any,
    headers: Dict[str, str],
    age_info: Dict[str, Any],
    sender_info: Dict[str, str],
) -> str:
    """Get base context template without email body."""
    return LLM_CONTEXT_TEMPLATE.format(
        user_name=user_info.name,
        user_email=user_info.email,
        user_family_name=user_info.name.split()[-1] if user_info.name else "",
        user_timezone=get_user_timezone(),
        today_date=format_datetime_for_model(age_info["now"]),
        subject=headers["subject"],
        to_header=headers["to"],
        from_header=headers["from"],
        received_age=age_info["age_str"],
        sender_category=sender_info["sender_category"],
        sender_domain=sender_info["domain"],
        domain_category=sender_info["domain_category"],
        email_body="{email_body}",
    )


def _truncate_email_body(email_body: str, context_template: str, tokenizer: Any) -> str:
    """Truncate email body to fit within token limits."""
    max_context_tokens = int(tokenizer.model_max_length * 0.5)
    template_tokens = len(tokenizer.encode(context_template.format(email_body="")))
    available_tokens = max(max_context_tokens - template_tokens - 200, 0)

    body_tokens = tokenizer.encode(email_body)
    if len(body_tokens) > available_tokens:
        truncated_tokens = body_tokens[:available_tokens]
        email_body = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        email_body += "\n\n[... Content truncated ...]"

    return email_body


async def _get_final_labels(msg_id: str, context: str, model: Any, tokenizer: Any) -> List[str]:
    """Get final normalized labels for the message."""
    try:
        label_list = await classify_email(context, model, tokenizer)
    except Exception as e:
        logger.error("LLM classification failed for message %s: %s", msg_id, str(e))
        return [f"{LLM_PARENT_LABEL}/error"]

    logger.info("Message %s raw LLM classification => %s", msg_id, label_list)
    final_labels = [apply_parent_label(label) for label in label_list]
    logger.info("Message %s normalized final labels => %s", msg_id, final_labels)

    for label in final_labels:
        if label.lower() == f"{PARENT_LABEL.lower()}/ignore":
            logger.info("Message %s has 'ignore' label => skipping other labels.", msg_id)
            return [label]

    return final_labels


async def classify_inbox_message(
    aiogoogle: Aiogoogle,
    msg_meta: Dict[str, Any],
    sla_data: SLAData,
    llm_model: Any,
    tokenizer: Any,
) -> Tuple[str, List[str]]:
    """
    Lightweight wrapper to fetch final labels for a message using the LLM-based classification.
    Returns (msg_id, final_labels).
    """
    msg_id = msg_meta["id"]
    logger.info("classify_inbox_message => msg_id=%s", msg_id)
    final_labels = await process_inbox_message(aiogoogle, msg_id, sla_data, llm_model, tokenizer)
    logger.info("Message %s classification complete => final_labels=%s", msg_id, final_labels)
    return msg_id, final_labels


async def apply_labels(aiogoogle: Aiogoogle, msg_id: str, labels: List[str]) -> None:
    logger.info("Applying labels to message %s => %s", msg_id, labels)
    label_ids = []
    for label_name in labels:
        logger.debug("Checking if label needs creation: %s", label_name)
        label_id = await get_or_create_label(aiogoogle, label_name)
        logger.debug("Label '%s' is ready with ID: %s", label_name, label_id)
        label_ids.append(label_id)

    logger.debug("Updating message %s with label IDs: %s", msg_id, label_ids)
    result = await update_message_labels(aiogoogle, msg_id, label_ids, labels)
    if result:
        logger.info(
            "Successfully applied labels to message %s =>\n%s",
            msg_id,
            "\n".join(f"  - {label}" for label in labels),
        )
    else:
        logger.warning("Failed to apply some or all labels to message %s", msg_id)


async def fetch_existing_labels(
    aiogoogle: Aiogoogle,
    gmail: Any,
    msg_meta: Dict[str, Any],
    all_labels_by_id: Dict[str, str],
) -> List[str]:
    msg_id = msg_meta["id"]
    existing_labels = msg_meta.get("labelIds", [])
    if not existing_labels:
        logger.info("Message %s has no labelIds in msg_meta; fetching metadata...", msg_id)
        full_msg = await get_message(aiogoogle, msg_id, format="metadata", metadata_headers=["labelIds"])
        existing_labels = full_msg.get("labelIds", [])
        logger.info("After fetching metadata, msg %s labelIds: %s", msg_id, existing_labels)

    if not existing_labels and msg_meta.get("threadId"):
        thread_id = msg_meta["threadId"]
        logger.info("Message %s still has no labelIds; checking thread %s", msg_id, thread_id)
        thread = await aiogoogle.as_user(
            gmail.users.threads.get(
                userId="me",
                id=thread_id,
                format="metadata",
                metadataHeaders=["labelIds"],
            )
        )
        thread_labels = thread.get("labels", []) or thread.get("labelIds", [])
        existing_labels = thread_labels
        existing_label_names = [all_labels_by_id.get(lbl_id, lbl_id) for lbl_id in existing_labels]
        logger.info("Message %s thread-level labels: %s", msg_id, existing_label_names)

    return existing_labels


async def process_inbox_folder(
    aiogoogle: Aiogoogle,
    llm_model: Any,
    tokenizer: Any,
    sla_data: SLAData,
    max_messages: Optional[int] = None,
) -> None:
    """Process messages in the inbox folder."""
    messages = await _get_inbox_messages(aiogoogle, max_messages)
    if not messages:
        return

    label_info = await _get_label_info(aiogoogle)
    if not label_info:
        return

    await _process_messages(
        aiogoogle=aiogoogle,
        messages=messages,
        label_info=label_info,
        llm_model=llm_model,
        tokenizer=tokenizer,
        sla_data=sla_data,
    )


async def _get_inbox_messages(
    aiogoogle: Aiogoogle,
    max_messages: Optional[int],
) -> List[Dict[str, Any]]:
    """Fetch messages from inbox."""
    logger.info("Fetching up to %d messages from [INBOX] ...", max_messages if max_messages else 0)
    messages = await list_messages(aiogoogle, "INBOX", max_messages)
    logger.info("Found %d messages in INBOX to process.", len(messages))
    return messages


async def _get_label_info(aiogoogle: Aiogoogle) -> Optional[Dict[str, Any]]:
    """Get Gmail label information."""
    gmail = await aiogoogle.discover("gmail", "v1")
    response = await aiogoogle.as_user(gmail.users.labels.list(userId="me"))

    all_labels_info = {label.get("name", ""): label.get("id") for label in response.get("labels", [])}
    all_labels_by_id = {v: k for k, v in all_labels_info.items() if v}
    logger.info("All Gmail labels => %s", list(all_labels_info.keys()))

    zola_label_info = {
        lbl["name"]: lbl["id"]
        for lbl in response.get("labels", [])
        if lbl.get("name", "").lower() == PARENT_LABEL.lower()
        or lbl.get("name", "").lower().startswith(f"{PARENT_LABEL.lower()}/")
    }
    logger.info("Detected zola or zola sub-labels => %s", list(zola_label_info.keys()))

    return {
        "all_labels_info": all_labels_info,
        "all_labels_by_id": all_labels_by_id,
        "zola_label_info": zola_label_info,
        "zola_labels": set(zola_label_info.values()),
        "gmail": gmail,
    }


async def _process_messages(
    aiogoogle: Aiogoogle,
    messages: List[Dict[str, Any]],
    label_info: Dict[str, Any],
    llm_model: Any,
    tokenizer: Any,
    sla_data: SLAData,
) -> None:
    """Process each inbox message."""
    logger.info("Beginning classification of INBOX messages ...")
    label_tasks = []
    stats = {"total": len(messages), "skipped": 0, "llm_processed": 0}

    with tqdm(total=stats["total"], desc="Inbox messages", unit="msg") as pbar:
        for msg_meta in messages:
            try:
                task = await _handle_message(
                    aiogoogle=aiogoogle,
                    msg_meta=msg_meta,
                    label_info=label_info,
                    llm_model=llm_model,
                    tokenizer=tokenizer,
                    sla_data=sla_data,
                    stats=stats,
                )
                if task:
                    label_tasks.append(task)
            except Exception as e:
                logger.error("Error processing message %s: %s", msg_meta["id"], e, exc_info=True)
                stats["skipped"] += 1
            finally:
                pbar.update(1)

        if label_tasks:
            logger.info("Waiting for %d label apply tasks to complete...", len(label_tasks))
            await asyncio.gather(*label_tasks)

    await _print_summary(aiogoogle, stats)


async def _handle_message(
    aiogoogle: Aiogoogle,
    msg_meta: Dict[str, Any],
    label_info: Dict[str, Any],
    llm_model: Any,
    tokenizer: Any,
    sla_data: SLAData,
    stats: Dict[str, int],
) -> Optional[asyncio.Task]:
    """Process a single message and return a label task if needed."""
    msg_id = msg_meta["id"]
    logger.info("Processing inbox message => ID=%s, labelIds=%s", msg_id, msg_meta.get("labelIds", []))

    existing_labels = await fetch_existing_labels(
        aiogoogle, label_info["gmail"], msg_meta, label_info["all_labels_by_id"]
    )

    if _has_zola_labels(existing_labels, label_info):
        stats["skipped"] += 1
        return None

    logger.info("No existing zola label found, classifying message %s with LLM ...", msg_id)
    new_msg_id, final_labels = await classify_inbox_message(aiogoogle, msg_meta, sla_data, llm_model, tokenizer)

    if not final_labels:
        logger.info("Message %s => no labels returned from classification, skipping label application.", new_msg_id)
        stats["skipped"] += 1
        return None

    logger.info("Message %s => final labels => %s", new_msg_id, final_labels)
    stats["llm_processed"] += 1
    return asyncio.create_task(apply_labels(aiogoogle, new_msg_id, final_labels))


def _has_zola_labels(existing_labels: List[str], label_info: Dict[str, Any]) -> bool:
    """Check if message already has Zola labels."""
    intersection = [lbl_id for lbl_id in existing_labels if lbl_id in label_info["zola_labels"]]
    if intersection:
        intersection_label_names = [label_info["all_labels_by_id"].get(lbl_id, lbl_id) for lbl_id in intersection]
        logger.info("SKIPPING message => already has zola label(s): %s", intersection_label_names)
        return True
    return False


async def _print_summary(aiogoogle: Aiogoogle, stats: Dict[str, int]) -> None:
    """Print processing summary."""
    logger.info("Finished processing INBOX messages.")
    user_info = await get_user_info(aiogoogle)
    summary_text = (
        f"[bold green]{user_info.shortname} ({user_info.email}) Summary[/bold green]\n\n"
        f"Total messages: {stats['total']}\n"
        f"LLM-processed: {stats['llm_processed']}\n"
        f"Skipped: {stats['skipped']}"
    )
    console.print(Panel(summary_text, border_style="green"))
