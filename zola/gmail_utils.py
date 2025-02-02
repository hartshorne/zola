import asyncio
import base64
import email.utils
import logging
import os
import re
import tempfile
import time
import zoneinfo
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiogoogle.client import Aiogoogle
from markitdown import MarkItDown

from zola.settings import ALL_LABELS, LLM_PARENT_LABEL, PARENT_LABEL

logger = logging.getLogger(__name__)

# Precompile the email regex pattern for efficiency.
EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9.+_-]+@([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*)"
)


class RateLimiter:
    """
    Asynchronous rate limiter that spreads out calls evenly, enforcing:
      * at most max_calls over 'period' seconds
      * i.e., a minimum delay of (period / max_calls) between consecutive calls
    """

    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self.min_interval = self.period / self.max_calls  # e.g. 1/100 = 0.01
        self.lock = asyncio.Lock()
        self.last_call_time: Optional[float] = None  # NOTE: Explicit type annotation.

    async def __aenter__(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if self.last_call_time is None:
                self.last_call_time = now
                return
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_call_time = time.monotonic()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# Create a module-wide rate limiter for ~100 calls per second.
rate_limiter = RateLimiter(100, 1.0)


async def list_messages(aiogoogle: Aiogoogle, label: str, max_messages: int = 1000) -> List[Dict[str, Any]]:
    """
    Retrieves up to max_messages message metadata for a given Gmail label.
    """
    messages = []
    gmail = await aiogoogle.discover("gmail", "v1")
    response = await aiogoogle.as_user(gmail.users.messages.list(userId="me", labelIds=[label], maxResults=500))
    if "messages" in response:
        messages.extend(response["messages"])
    while "nextPageToken" in response and len(messages) < max_messages:
        page_token = response["nextPageToken"]
        response = await aiogoogle.as_user(
            gmail.users.messages.list(
                userId="me",
                labelIds=[label],
                pageToken=page_token,
                maxResults=500,
            )
        )
        messages.extend(response.get("messages", []))
    return messages[:max_messages]


async def get_message(
    aiogoogle: Aiogoogle,
    msg_id: str,
    format: str = "full",
    metadata_headers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Retrieves a Gmail message by ID.
    """
    gmail = await aiogoogle.discover("gmail", "v1")
    async with rate_limiter:
        params = {"userId": "me", "id": msg_id, "format": format}
        if format == "metadata" and metadata_headers:
            params["metadataHeaders"] = metadata_headers
        message = await aiogoogle.as_user(gmail.users.messages.get(**params))
    return message


async def get_attachment(aiogoogle: Aiogoogle, msg_id: str, attachment_id: str) -> bytes:
    """
    Downloads an attachment from a Gmail message.
    """
    gmail = await aiogoogle.discover("gmail", "v1")
    async with rate_limiter:
        attachment = await aiogoogle.as_user(
            gmail.users.messages.attachments.get(userId="me", messageId=msg_id, id=attachment_id)
        )
    data = attachment.get("data", "")
    return decode_base64url(data)


def get_message_attachments(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts attachment metadata from a full Gmail message.
    Returns a list of dictionaries for each attachment with:
      - filename: The attachment's filename (or 'untitled' if not provided)
      - mimeType: The MIME type
      - size: Size in bytes
      - attachmentId: ID required to download the attachment
    """
    attachments: List[Dict[str, Any]] = []

    def process_parts(parts: List[Dict[str, Any]]) -> None:
        for part in parts:
            body = part.get("body", {})
            if body.get("attachmentId") or part.get("filename"):
                attachments.append(
                    {
                        "filename": part.get("filename") or "untitled",
                        "mimeType": part.get("mimeType"),
                        "size": body.get("size", 0),
                        "attachmentId": body.get("attachmentId"),
                    }
                )
            if "parts" in part:
                process_parts(part["parts"])

    payload = message.get("payload", {})
    if "parts" in payload:
        process_parts(payload["parts"])
    return attachments


async def get_or_create_label(aiogoogle: Aiogoogle, label_name: str) -> str:
    """
    Ensures a Gmail label exists and returns its ID.
    Creates nested labels under 'zola' parent label.
    Only creates labels that are defined in ALL_LABELS.
    """
    if label_name not in ALL_LABELS and not label_name == PARENT_LABEL:
        raise ValueError(f"Invalid label name: {label_name}")

    gmail = await aiogoogle.discover("gmail", "v1")

    # First check if the label already exists (case-insensitive)
    response = await aiogoogle.as_user(gmail.users.labels.list(userId="me"))
    for label in response.get("labels", []):
        if label.get("name", "").lower() == label_name.lower():
            logger.debug("Found existing label '%s' with ID: %s", label_name, label.get("id"))
            return label.get("id")

    # If it's a nested label, ensure parent exists first
    if label_name.startswith(f"{PARENT_LABEL}/"):
        parent_name = PARENT_LABEL
        parent_label_id = None
        for label in response.get("labels", []):
            if label.get("name", "").lower() == parent_name.lower():
                parent_label_id = label.get("id")
                logger.debug(
                    f"Found existing parent label '{PARENT_LABEL}' with ID: %s",
                    parent_label_id,
                )
                break
        if not parent_label_id:
            logger.info("Creating parent label: %s", parent_name)
            try:
                parent_response = await aiogoogle.as_user(
                    gmail.users.labels.create(
                        userId="me",
                        json={
                            "name": parent_name,
                            "labelListVisibility": "labelShow",
                            "messageListVisibility": "show",
                            "type": "user",
                            "color": {
                                "backgroundColor": "#666666",
                                "textColor": "#ffffff",
                            },
                        },
                    )
                )
                parent_label_id = parent_response.get("id")
                logger.info("Created parent label 'zola' with ID: %s", parent_label_id)
            except Exception as e:
                if "Label name exists or conflicts" in str(e):
                    # Try to fetch the existing label again
                    response = await aiogoogle.as_user(gmail.users.labels.list(userId="me"))
                    for label in response.get("labels", []):
                        if label.get("name", "").lower() == parent_name.lower():
                            parent_label_id = label.get("id")
                            logger.debug("Found parent label after conflict: %s", parent_label_id)
                            break
                if not parent_label_id:
                    raise

    # Create the new label
    logger.info("Creating new Gmail label: %s", label_name)
    try:
        create_response = await aiogoogle.as_user(
            gmail.users.labels.create(
                userId="me",
                json={
                    "name": label_name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                    "type": "user",
                    "color": {"backgroundColor": "#666666", "textColor": "#ffffff"},
                },
            )
        )
        new_label_id = create_response.get("id")
        logger.info("Successfully created new label '%s' with ID: %s", label_name, new_label_id)
        return new_label_id
    except Exception as e:
        if "Label name exists or conflicts" in str(e):
            # Try to fetch the existing label one more time
            response = await aiogoogle.as_user(gmail.users.labels.list(userId="me"))
            for label in response.get("labels", []):
                if label.get("name", "").lower() == label_name.lower():
                    logger.debug(
                        "Found label after conflict: %s with ID: %s",
                        label_name,
                        label.get("id"),
                    )
                    return label.get("id")
        raise


async def update_message_labels(
    aiogoogle: Aiogoogle,
    msg_id: str,
    label_ids: List[str],
    label_names: List[str],
    max_retries: int = 5,
) -> bool:
    """
    Adds a list of label IDs to a Gmail message, with exponential backoff retry.
    Returns True if labels were successfully applied, False otherwise.
    """
    gmail = await aiogoogle.discover("gmail", "v1")
    backoff = 1  # initial backoff in seconds
    for attempt in range(max_retries):
        logger.debug(
            "Attempting to update message %s with label IDs %s (attempt %d/%d)",
            msg_id,
            label_ids,
            attempt + 1,
            max_retries,
        )
        try:
            if not label_ids:
                logger.warning("No label IDs provided for message %s", msg_id)
                return False

            response = await aiogoogle.as_user(
                gmail.users.messages.modify(
                    userId="me",
                    id=msg_id,
                    json={"addLabelIds": label_ids, "removeLabelIds": []},
                )
            )

            # Log the response details
            if response and "labelIds" in response:
                applied_labels = set(response["labelIds"])
                requested_labels = set(label_ids)
                if requested_labels.issubset(applied_labels):
                    logger.debug(
                        "Gmail API confirmed labels were applied to message %s:\nRequested: %s\nCurrent labels: %s",
                        msg_id,
                        label_names,
                        response["labelIds"],
                    )
                    return True
                else:
                    missing_labels = requested_labels - applied_labels
                    logger.warning(
                        "Some requested labels were not applied to message %s:\nMissing: %s\nCurrent labels: %s",
                        msg_id,
                        [lid for lid in label_ids if lid in missing_labels],
                        response["labelIds"],
                    )
                    return False
            else:
                logger.warning(
                    "Unexpected response format from Gmail API for message %s: %s",
                    msg_id,
                    response,
                )
                return False

        except Exception as e:
            logger.error(
                "Failed to update message %s with labels %s (IDs: %s) on attempt %d: %s",
                msg_id,
                label_names,
                label_ids,
                attempt + 1,
                e,
                exc_info=True,
            )
            if attempt == max_retries - 1:
                return False
            await asyncio.sleep(backoff)
            backoff *= 2

    return False


def parse_date(date_str: str) -> datetime:
    """
    Parses an RFC2822 date string into a datetime object.
    """
    dt = email.utils.parsedate_to_datetime(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def extract_email_domain(header_value: str) -> str:
    """
    Extracts the email domain from a header string.
    """
    text_outside_brackets = re.sub(r"<[^>]*>", "", header_value)
    if text_outside_brackets.count("@") > 1:
        return "unknown"
    match = EMAIL_REGEX.search(header_value)
    return match.group(1).lower() if match else "unknown"


def extract_message_part(msg: Dict[str, Any]) -> str:
    """
    Recursively walk through the email parts to find message body.
    """
    if not msg:
        return ""
    mime_type = msg.get("mimeType", "")
    if mime_type.startswith("text/"):
        if msg.get("filename"):
            return ""
        body_data = msg.get("body", {}).get("data")
        if body_data:
            try:
                return decode_base64url(body_data).decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning("Failed to decode message body: %s", e)
                return ""
    if "parts" in msg:
        for part in msg["parts"]:
            if part.get("mimeType") == "text/plain" and not part.get("filename"):
                content = extract_message_part(part)
                if content:
                    return content
        for part in msg["parts"]:
            if part.get("mimeType") == "text/html" and not part.get("filename"):
                content = extract_message_part(part)
                if content:
                    return content
        for part in msg["parts"]:
            if not part.get("filename"):
                content = extract_message_part(part)
                if content:
                    return content
    return ""


async def get_email_body_markdown(
    aiogoogle: Aiogoogle,
    message: Dict[str, Any],
    include_attachment_content: bool = True,
) -> str:
    """
    Extracts the email body and attachments, converting everything to Markdown.
    """
    payload = message.get("payload", {})
    if not payload:
        logger.debug("No payload found in message; falling back to snippet")
        return message.get("snippet", "").strip()

    logger.debug(
        "Message payload MIME type: %s; parts present: %s; headers: %s",
        payload.get("mimeType", "unknown"),
        bool(payload.get("parts")),
        payload.get("headers", []),
    )

    content = extract_message_part(payload)
    if not content:
        logger.debug("No content extracted, falling back to snippet")
        content = message.get("snippet", "").strip()
        if not content:
            return ""

    # If the content is HTML, convert it to Markdown using MarkItDown.
    if payload.get("mimeType", "").startswith("text/html"):
        logger.debug("Converting HTML content using MarkItDown")
        md = MarkItDown()
        # Write the HTML content to a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        result = md.convert(tmp_path)
        if result and result.text_content:
            content = result.text_content.strip()
        os.remove(tmp_path)  # Remove the temporary file.

    content_parts = [content]
    attachments = get_message_attachments(message)
    if attachments:
        content_parts.append("\n## Attachments")
        supported_mime_types = {
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        for attachment in attachments:
            content_parts.append(
                f"- **{attachment['filename']}** ({attachment['mimeType']}, {attachment['size']} bytes)"
            )
            if include_attachment_content and (
                attachment["mimeType"].startswith("text/") or attachment["mimeType"] in supported_mime_types
            ):
                try:
                    logger.debug(
                        "Processing attachment '%s' with MIME type '%s'",
                        attachment["filename"],
                        attachment["mimeType"],
                    )
                    attachment_data = None
                    # Look for inline attachment data.
                    if "parts" in payload:
                        for part in payload["parts"]:
                            if part.get("filename") == attachment["filename"] and part.get("body", {}).get("data"):
                                logger.debug(
                                    "Found inline attachment data for '%s'",
                                    attachment["filename"],
                                )
                                attachment_data = decode_base64url(part["body"]["data"])
                                break
                    if not attachment_data and attachment.get("attachmentId"):
                        logger.debug(
                            "No inline data found; downloading attachment '%s' with ID '%s'",
                            attachment["filename"],
                            attachment.get("attachmentId"),
                        )
                        attachment_data = await get_attachment(aiogoogle, message["id"], attachment["attachmentId"])
                    if attachment_data:
                        logger.debug(
                            "Attachment data length for '%s': %d bytes",
                            attachment["filename"],
                            len(attachment_data),
                        )
                        # Create a temporary file for the attachment.
                        _, ext = os.path.splitext(attachment["filename"])
                        if not ext:
                            if attachment["mimeType"] == "application/pdf":
                                ext = ".pdf"
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                            tmp.write(attachment_data)
                            tmp.flush()
                            temp_filename = tmp.name
                        logger.debug(
                            "Temporary file created for '%s': %s",
                            attachment["filename"],
                            temp_filename,
                        )
                        md = MarkItDown()
                        logger.debug(
                            "Starting MarkItDown conversion for attachment '%s' using tempfile",
                            attachment["filename"],
                        )
                        result = md.convert(temp_filename)
                        logger.debug(
                            "MarkItDown conversion result for '%s': %s",
                            attachment["filename"],
                            result,
                        )
                        converted_text = result.text_content.strip() if (result and result.text_content) else ""
                        logger.debug(
                            "Converted text length for attachment '%s': %d",
                            attachment["filename"],
                            len(converted_text),
                        )
                        if converted_text:
                            content_parts.append(f"\n```\n{converted_text}\n```")
                        os.remove(temp_filename)
                        logger.debug("Temporary file '%s' removed", temp_filename)
                    else:
                        logger.debug(
                            "No attachment data found for '%s'",
                            attachment["filename"],
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to convert attachment %s (%s): %s",
                        attachment["filename"],
                        attachment["mimeType"],
                        e,
                    )

    final_content = "\n\n".join(content_parts).strip()
    logger.debug(
        "Final Markdown content (length %d): %s",
        len(final_content),
        final_content,
    )
    return final_content


async def remove_empty_labels(aiogoogle: Aiogoogle) -> None:
    """
    First creates the system-level labels, then checks all zola labels and removes any that have no messages.
    Preserves the special hierarchy labels even if empty.
    """
    gmail = await aiogoogle.discover("gmail", "v1")

    # First, ensure the system-level labels exist
    logger.info("Creating system-level labels if needed")
    await asyncio.gather(
        get_or_create_label(aiogoogle, PARENT_LABEL),
        get_or_create_label(aiogoogle, LLM_PARENT_LABEL),
    )

    # Then proceed with cleaning up empty labels
    response = await aiogoogle.as_user(gmail.users.labels.list(userId="me"))
    preserved_labels = {
        PARENT_LABEL.lower(),
        LLM_PARENT_LABEL.lower(),
    }
    for label in response.get("labels", []):
        name = label.get("name", "")
        if name.lower().startswith(f"{PARENT_LABEL.lower()}/"):
            if name.lower() in preserved_labels:
                continue
            label_id = label.get("id")
            messages_response = await aiogoogle.as_user(
                gmail.users.messages.list(userId="me", labelIds=[label_id], maxResults=1)
            )
            if not messages_response.get("messages"):
                logger.info("Removing empty label: %s", name)
                try:
                    await aiogoogle.as_user(gmail.users.labels.delete(userId="me", id=label_id))
                except Exception as e:
                    logger.warning("Failed to delete empty label %s: %s", name, e)


def get_user_timezone() -> str:
    """
    Get the user's timezone. Tries multiple methods:
      1. TZ environment variable
      2. System timezone from /etc/timezone
      3. Tries /etc/localtime symlink
      4. Falls back to UTC if determination fails
    """
    tz = os.environ.get("TZ")
    if tz:
        try:
            zoneinfo.ZoneInfo(tz)
            return tz
        except zoneinfo.ZoneInfoNotFoundError:
            pass
    try:
        with open("/etc/timezone") as f:
            tz_candidate = f.read().strip()
            if tz_candidate:
                zoneinfo.ZoneInfo(tz_candidate)
                return tz_candidate
    except (FileNotFoundError, zoneinfo.ZoneInfoNotFoundError):
        pass
    try:
        localtime = Path("/etc/localtime")
        if localtime.exists():
            target = os.path.realpath(localtime)
            if "zoneinfo" in target:
                tz_candidate = target.split("zoneinfo/")[-1].split("zoneinfo.default/")[-1]
                tz_candidate = tz_candidate.lstrip("/")
                if tz_candidate in zoneinfo.available_timezones():
                    return tz_candidate
    except (OSError, zoneinfo.ZoneInfoNotFoundError):
        pass
    logger.warning("Could not determine user timezone, falling back to UTC")
    return "UTC"


def convert_to_user_timezone(dt: datetime) -> datetime:
    """
    Convert a datetime to the user's timezone.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    user_tz = zoneinfo.ZoneInfo(get_user_timezone())
    return dt.astimezone(user_tz)


def format_datetime_for_model(dt: datetime) -> str:
    """
    Format a datetime in a consistent way for the model context.
    """
    local_dt = convert_to_user_timezone(dt)
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def decode_base64url(data: str) -> bytes:
    """
    Decodes a base64url-encoded string, adding any missing padding.
    """
    if not data:
        return b""
    data = data.replace("-", "+").replace("_", "/")
    missing_padding = len(data) % 4
    if missing_padding:
        data += "=" * (4 - missing_padding)
    return base64.b64decode(data)


class TempFileManager:
    def __init__(self, suffix: str = None):
        self.suffix = suffix
        self.temp_file = None

    def __enter__(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+b", suffix=self.suffix, delete=False)
        return self.temp_file.name

    def __exit__(self, *_):
        if self.temp_file:
            self.temp_file.close()
            if os.path.exists(self.temp_file.name):
                os.remove(self.temp_file.name)
