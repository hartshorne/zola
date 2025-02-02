import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from aiogoogle.client import Aiogoogle
from humanize import naturaldelta
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

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


class TimeBucket(NamedTuple):
    """Represents a time bucket for SLA calculations with its weight."""

    name: str
    duration: timedelta
    weight: float


class SLAData:
    """Handles SLA data collection and calculation with time‐weighted buckets."""

    BUCKETS = [
        TimeBucket("hour", timedelta(hours=1), 1.0),
        TimeBucket("day", timedelta(days=1), 0.7),
        TimeBucket("week", timedelta(days=7), 0.5),
        TimeBucket("month", timedelta(days=30), 0.3),
    ]

    def __init__(self) -> None:
        self.sender_data: Dict[str, Dict[str, List[float]]] = {}
        self.domain_data: Dict[str, Dict[str, List[float]]] = {}
        self.known_senders = set()
        self.known_domains = set()

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
            self.sender_data[sender] = {bucket.name: [] for bucket in self.BUCKETS}
        if domain not in self.domain_data:
            self.domain_data[domain] = {bucket.name: [] for bucket in self.BUCKETS}
        for bucket in self.BUCKETS:
            if age <= bucket.duration:
                self.sender_data[sender][bucket.name].append(response_time)
                self.domain_data[domain][bucket.name].append(response_time)
        bucket_names = [b.name for b in self.BUCKETS if age <= b.duration]
        logger.debug(
            f"Added response time:\n  Sender: {sender}\n  Domain: {domain}\n"
            f"  Response Time: {naturaldelta(response_time)}\n  Age: {naturaldelta(age)}\n"
            f"  Added to buckets: {bucket_names}"
        )

    def _calculate_weighted_sla(self, bucket_data: Dict[str, List[float]]) -> float:
        total_responses = sum(len(bucket_data.get(bucket.name, [])) for bucket in self.BUCKETS)
        if total_responses == 0:
            return float("inf")
        weighted_sum = 0.0
        total_weight = 0.0
        for bucket in self.BUCKETS:
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

        def build_table(data: Dict[str, Dict[str, List[float]]], known: set, title: str) -> Table:
            table = Table(title=title, show_header=True, header_style="bold")
            table.add_column(title.split()[0])
            for bucket in self.BUCKETS:
                table.add_column(f"{bucket.name}\n(w={bucket.weight:.1f})")
            table.add_column("Overall SLA")
            table.add_column("Category")
            for key in sorted(known):
                row = [key]
                buckets = data.get(key, {})
                if buckets:
                    total_responses = sum(len(times) for times in buckets.values())
                    for bucket in self.BUCKETS:
                        times = buckets.get(bucket.name, [])
                        if times:
                            avg = sum(times) / len(times)
                            resp_weight = len(times) / total_responses
                            row.append(f"{len(times)} resp\n{naturaldelta(avg)}\n(w={bucket.weight * resp_weight:.2f})")
                        else:
                            row.append("-")
                    overall_sla = self._calculate_weighted_sla(buckets)
                    row.append(naturaldelta(overall_sla) if overall_sla != float("inf") else "no data")
                    row.append(self._get_sla_category(overall_sla, True))
                else:
                    row.extend(["-"] * (len(self.BUCKETS) + 2))
                table.add_row(*row)
            return table

        sender_table = build_table(self.sender_data, self.known_senders, "Sender Response Times")
        domain_table = build_table(self.domain_data, self.known_domains, "Domain Response Times")
        console = Console()
        console.print("\n")
        console.print(sender_table)
        console.print("\n")
        console.print(domain_table)
        console.print("\n")


async def get_user_info(aiogoogle: Aiogoogle) -> Tuple[str, str]:
    if hasattr(aiogoogle, "_user_info"):
        return aiogoogle._user_info  # type: ignore

    accounts_path = Path("secrets/accounts.json")
    if accounts_path.exists():
        with accounts_path.open("r") as f:
            accounts = json.load(f)
    else:
        accounts = {}

    gmail = await aiogoogle.discover("gmail", "v1")
    profile = await aiogoogle.as_user(gmail.users.getProfile(userId="me"))
    my_email = profile.get("emailAddress", "me")
    for _, account_info in accounts.items():
        if account_info.get("email") == my_email:
            user_info = (account_info.get("name", "User"), my_email)
            aiogoogle._user_info = user_info  # Cache it
            return user_info
    logger.info("No account found for %s. Please enter your name:", my_email)
    user_name = input("Enter your name: ").strip()
    accounts["work"] = {"email": my_email, "name": user_name}
    with accounts_path.open("w") as f:
        json.dump(accounts, f, indent=2)
    logger.info("Saved user information to accounts.json")
    user_info = (user_name, my_email)
    aiogoogle._user_info = user_info
    return user_info


async def process_sent_message(
    aiogoogle: Aiogoogle,
    msg_meta: Dict[str, Any],
    pbar: Any,
    sla_data: SLAData,
    my_email: str,
) -> None:
    msg_id = msg_meta["id"]
    logger.info("Processing sent message with id: %s", msg_id)
    try:
        gmail = await aiogoogle.discover("gmail", "v1")
        message = await aiogoogle.as_user(
            gmail.users.messages.get(
                userId="me",
                id=msg_id,
                format="metadata",
                metadataHeaders=["date", "in-reply-to", "to", "from"],
            )
        )
        headers = message.get("payload", {}).get("headers", [])
        in_reply_to = next((h["value"] for h in headers if h["name"].lower() == "in-reply-to"), None)
        sent_date_str = next((h["value"] for h in headers if h["name"].lower() == "date"), None)
        if not sent_date_str:
            logger.info("Message %s does not contain a Date header; skipping.", msg_id)
            return
        sent_date = parse_date(sent_date_str)
        if in_reply_to:
            thread_id = message.get("threadId")
            thread = await aiogoogle.as_user(
                gmail.users.threads.get(
                    userId="me",
                    id=thread_id,
                    format="metadata",
                    metadataHeaders=["date", "from"],
                )
            )
            thread_msgs = thread.get("messages", [])
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
            if candidates:
                candidate_date, candidate = max(candidates, key=lambda x: x[0])
                candidate_headers = candidate.get("payload", {}).get("headers", [])
                from_header = next((h["value"] for h in candidate_headers if h["name"].lower() == "from"), "")
                sender = from_header
                domain = extract_email_domain(from_header)
                reply_time = (sent_date - candidate_date).total_seconds()
                sla_data.add_response_time(sender, domain, reply_time, sent_date)
                logger.info(
                    "Processed sent message %s: computed reply time %.2f seconds for sender %s (domain %s).",
                    msg_id,
                    reply_time,
                    sender,
                    domain,
                )
        else:
            to_header = next((h["value"] for h in headers if h["name"].lower() == "to"), "")
            sender = to_header
            domain = extract_email_domain(to_header)
            default_sla = DEFAULT_SLA_SECONDS
            sla_data.add_response_time(sender, domain, default_sla, sent_date)
            logger.info(
                "Processed sent message %s: using default SLA of %.2f seconds for new conversation with recipient %s (domain %s).",
                msg_id,
                default_sla,
                sender,
                domain,
            )
    except Exception as e:
        logger.warning("Error processing sent message %s: %s", msg_id, str(e))
    finally:
        pbar.update(1)


async def process_sent_folder(
    aiogoogle: Aiogoogle,
    sla_data: SLAData,
    max_messages: int,
) -> None:
    messages = await list_messages(aiogoogle, "SENT", max_messages)
    logger.info("Starting to process %d sent messages.", len(messages))
    with tqdm(total=len(messages), desc="Sent messages", unit="msg") as pbar:
        user_name, my_email = await get_user_info(aiogoogle)
        tasks = [
            asyncio.create_task(process_sent_message(aiogoogle, msg_meta, pbar, sla_data, my_email))
            for msg_meta in messages
        ]
        if tasks:
            await asyncio.gather(*tasks)
    logger.info("Finished processing sent messages.")


def normalize_label(label: str) -> Optional[str]:
    lower_label = label.lower()
    if lower_label in STANDARD_LABELS:
        return lower_label
    return label.lower().replace("_", " ")


async def process_inbox_message(
    aiogoogle: Aiogoogle,
    msg_id: str,
    sla_data: SLAData,
    model: Any,
    tokenizer: Any,
) -> List[str]:
    try:
        message = await get_message(
            aiogoogle,
            msg_id,
            format="full",
            metadata_headers=["From", "To", "Subject", "Date"],
        )
        if message is None:
            logger.warning("Failed to fetch message %s", msg_id)
            return [f"{LLM_PARENT_LABEL}/error"]

        headers = message.get("payload", {}).get("headers", [])
        subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "")
        to_header = next((h["value"] for h in headers if h["name"].lower() == "to"), "")
        from_header = next((h["value"] for h in headers if h["name"].lower() == "from"), "")
        date_header = next((h["value"] for h in headers if h["name"].lower() == "date"), "")
        now = datetime.now(timezone.utc)
        if date_header:
            email_datetime = parse_date(date_header)
            time_diff = now - email_datetime
            age_str = naturaldelta(time_diff)
        else:
            email_datetime = now
            age_str = "unknown age"
        sender = from_header
        sender_domain = extract_email_domain(from_header)
        sender_category, domain_category = sla_data.get_sla_categories(sender, sender_domain)
        email_body_md = await get_email_body_markdown(aiogoogle, message)
        user_name, user_email = await get_user_info(aiogoogle)

        context_template = LLM_CONTEXT_TEMPLATE
        context_without_body = context_template.format(
            user_name=user_name,
            user_email=user_email,
            user_family_name=user_name.split()[-1] if user_name else "",
            user_timezone=get_user_timezone(),
            today_date=format_datetime_for_model(now),
            subject=subject,
            to_header=to_header,
            from_header=from_header,
            received_age=age_str,
            sender_category=sender_category,
            sender_domain=sender_domain,
            domain_category=domain_category,
            email_body="",
        )
        max_context_tokens = int(tokenizer.model_max_length * 0.5)
        template_tokens = len(tokenizer.encode(context_without_body))
        available_tokens = max(max_context_tokens - template_tokens - 200, 0)
        email_body_tokens = tokenizer.encode(email_body_md)
        if len(email_body_tokens) > available_tokens:
            logger.info(
                "Truncating email body from %d to %d tokens",
                len(email_body_tokens),
                available_tokens,
            )
            truncated_tokens = email_body_tokens[:available_tokens]
            email_body_md = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            email_body_md += "\n\n[... Content truncated to fit model context ...]"

        context = context_template.format(
            user_name=user_name,
            user_email=user_email,
            user_family_name=user_name.split()[-1] if user_name else "",
            user_timezone=get_user_timezone(),
            today_date=format_datetime_for_model(now),
            subject=subject,
            to_header=to_header,
            from_header=from_header,
            received_age=age_str,
            sender_category=sender_category,
            sender_domain=sender_domain,
            domain_category=domain_category,
            email_body=email_body_md,
        )

        context_tokens = len(tokenizer.encode(context))
        logger.debug(
            "Context prepared for message %s: %d tokens (%.1f%% of model's limit)",
            msg_id,
            context_tokens,
            (context_tokens / tokenizer.model_max_length) * 100,
        )
        logger.debug(f"Full context for message {msg_id}:\n{'-' * 40}\n{context}\n{'-' * 40}")
        logger.debug("Classifying message %s with LLM...", msg_id)
        try:
            label_list = await classify_email(context, model, tokenizer)
        except Exception as e:
            logger.error("LLM classification failed for message %s: %s", msg_id, str(e))
            return [f"{LLM_PARENT_LABEL}/error"]

        logger.debug(f"Raw LLM response for message %s:\n{'-' * 40}\n{label_list}\n{'-' * 40}")
        logger.debug("Raw LLM classification for %s: %s", msg_id, label_list)

        # --- Normalization of labels ---
        final_labels = []
        for label in label_list:
            # If label doesn't already start with the parent label, add it.
            if not label.lower().startswith(f"{PARENT_LABEL.lower()}/"):
                label = f"{PARENT_LABEL}/{label}"
            # Remove duplicate parent label if present
            double_prefix = f"{PARENT_LABEL}/{PARENT_LABEL}/"
            if label.startswith(double_prefix):
                label = label.replace(double_prefix, f"{PARENT_LABEL}/", 1)
            final_labels.append(label)
        return final_labels
    except Exception as e:
        logger.error("Error processing message %s: %s", msg_id, str(e), exc_info=True)
        return [f"{LLM_PARENT_LABEL}/error"]


async def apply_labels(aiogoogle: Aiogoogle, msg_id: str, labels: List[str]) -> None:
    label_ids = []
    for label_name in labels:
        logger.debug("Checking if label exists: %s", label_name)
        label_id = await get_or_create_label(aiogoogle, label_name)
        logger.info("Label '%s' ready with ID: %s", label_name, label_id)
        label_ids.append(label_id)
    logger.debug("Applying label IDs to message %s: %s", msg_id, label_ids)
    result = await update_message_labels(aiogoogle, msg_id, label_ids, labels)
    if result:
        logger.info(
            "Successfully applied labels to message %s:\n%s",
            msg_id,
            "\n".join(f"  - {label}" for label in labels),
        )
    else:
        logger.warning("Failed to apply some or all labels to message %s", msg_id)


async def classify_inbox_message(
    aiogoogle: Aiogoogle,
    msg_meta: Dict[str, Any],
    sla_data: SLAData,
    llm_model: Any,
    tokenizer: Any,
) -> Tuple[str, List[str]]:
    msg_id = msg_meta["id"]
    final_labels = await process_inbox_message(aiogoogle, msg_id, sla_data, llm_model, tokenizer)
    return msg_id, final_labels


async def process_inbox_folder(
    aiogoogle: Aiogoogle,
    llm_model: Any,
    tokenizer: Any,
    sla_data: SLAData,
    max_messages: Optional[int] = None,
) -> None:
    messages = await list_messages(aiogoogle, "INBOX", max_messages)
    logger.info("Starting to process %d inbox messages.", len(messages))
    response = await aiogoogle.as_user((await aiogoogle.discover("gmail", "v1")).users.labels.list(userId="me"))
    zola_labels = {
        label["id"] for label in response.get("labels", []) if label.get("name", "").startswith(f"{PARENT_LABEL}/")
    }
    logger.debug("Found %d existing zola labels", len(zola_labels))
    label_tasks = []
    with tqdm(total=len(messages), desc="Inbox messages", unit="msg") as pbar:
        for msg_meta in messages:
            try:
                existing_labels = msg_meta.get("labelIds", [])
                if any(label_id in zola_labels for label_id in existing_labels):
                    logger.debug("Skipping message %s - already has zola labels", msg_meta["id"])
                    pbar.update(1)
                    continue
                msg_id, final_labels = await classify_inbox_message(aiogoogle, msg_meta, sla_data, llm_model, tokenizer)
                pbar.update(1)
                if final_labels:
                    label_task = asyncio.create_task(apply_labels(aiogoogle, msg_id, final_labels))
                    label_tasks.append(label_task)
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                pbar.update(1)
        if label_tasks:
            logger.info("Waiting for %d label tasks to complete...", len(label_tasks))
            await asyncio.gather(*label_tasks)
    logger.info("Finished processing inbox messages.")
