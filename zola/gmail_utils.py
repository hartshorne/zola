from __future__ import annotations

import asyncio
import base64
import contextlib
import email.utils
import logging
import os
import re
import tempfile
import zoneinfo
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import tenacity
from aiogoogle.client import Aiogoogle
from aiolimiter import AsyncLimiter
from markitdown import MarkItDown

# Local application imports
from zola.settings import ALL_LABELS, LLM_PARENT_LABEL, PARENT_LABEL

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9.+_-]+@([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,})")

# Create a module-wide async limiter for ~100 calls per second
rate_limiter = AsyncLimiter(100, 1.0)


async def get_gmail_service(aiogoogle: Aiogoogle) -> Any:
    """Return the Gmail API service."""
    return await aiogoogle.discover("gmail", "v1")


def decode_base64url(data: str) -> bytes:
    """Decode a base64url-encoded string, adding any missing padding."""
    if not data:
        return b""
    data = data.replace("-", "+").replace("_", "/")
    missing_padding = len(data) % 4
    if missing_padding:
        data += "=" * (4 - missing_padding)
    return base64.b64decode(data)


def extract_text_content(msg: Dict[str, Any]) -> str:
    """Recursively extract text content from an email message part."""
    if not msg:
        return ""
    mime_type = msg.get("mimeType", "")
    if mime_type.startswith("text/") and not msg.get("filename"):
        body_data = msg.get("body", {}).get("data")
        if body_data:
            try:
                return decode_base64url(body_data).decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning("Failed to decode message body: %s", e)
                return ""
    for part in msg.get("parts", []):
        content = extract_text_content(part)
        if content:
            return content
    return ""


def extract_email_domain(header_value: str) -> str:
    """Extract the email domain from a header string."""
    text_without_brackets = re.sub(r"<[^>]*>", "", header_value)
    if text_without_brackets.count("@") > 1:
        return "unknown"
    match = EMAIL_REGEX.search(header_value)
    if not match:
        return "unknown"
    return match.group(1).lower()


def parse_date(date_str: str) -> datetime:
    """Parse an RFC2822 date string into a datetime object (UTC if naive)."""
    dt = email.utils.parsedate_to_datetime(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_user_timezone() -> str:
    """Get the user's timezone from environment or system files; fall back to UTC."""
    tz = os.environ.get("TZ")
    if tz:
        try:
            zoneinfo.ZoneInfo(tz)
            return tz
        except zoneinfo.ZoneInfoNotFoundError:
            pass

    try:
        timezone_file = Path("/etc/timezone")
        if timezone_file.exists():
            tz_candidate = timezone_file.read_text().strip()
            if tz_candidate:
                zoneinfo.ZoneInfo(tz_candidate)
                return tz_candidate
    except zoneinfo.ZoneInfoNotFoundError:
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
    """Convert a datetime to the user's local timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    user_tz = zoneinfo.ZoneInfo(get_user_timezone())
    return dt.astimezone(user_tz)


def format_datetime_for_model(dt: datetime) -> str:
    """Format a datetime consistently for model context."""
    local_dt = convert_to_user_timezone(dt)
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


@contextlib.contextmanager
def temporary_file(suffix: Optional[str] = None) -> str:
    """
    Synchronous context manager for a temporary file that cleans up after use.
    Returns the path of the temporary file.
    """
    f = tempfile.NamedTemporaryFile(mode="w+b", suffix=suffix, delete=False)
    try:
        yield f.name
    finally:
        try:
            f.close()
        except Exception:
            pass
        temp_path = Path(f.name)
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning("Failed to remove temporary file %s: %s", temp_path, e)


@asynccontextmanager
async def async_temporary_file(suffix: Optional[str] = None) -> str:
    """
    Asynchronous context manager for a temporary file.
    It creates a temporary file (using a thread pool for blocking I/O),
    yields its path, and ensures cleanup on exit.
    """

    def create_temp():
        f = tempfile.NamedTemporaryFile(mode="w+b", suffix=suffix, delete=False)
        return f.name, f

    tmp_path, f = await asyncio.to_thread(create_temp)
    try:
        yield tmp_path
    finally:
        await asyncio.to_thread(f.close)

        def unlink_temp():
            try:
                Path(tmp_path).unlink()
            except Exception as e:
                logger.warning("Failed to remove temporary file %s: %s", tmp_path, e)

        await asyncio.to_thread(unlink_temp)


async def list_messages(
    aiogoogle: Aiogoogle,
    label: str,
    max_messages: int = 1000,
) -> List[Dict[str, Any]]:
    """Retrieve up to max_messages from Gmail under a particular label."""
    messages: List[Dict[str, Any]] = []
    gmail = await get_gmail_service(aiogoogle)

    async with rate_limiter:
        response = await aiogoogle.as_user(gmail.users.messages.list(userId="me", labelIds=[label], maxResults=500))

    if "messages" in response:
        messages.extend(response["messages"])

    while "nextPageToken" in response and len(messages) < max_messages:
        page_token = response["nextPageToken"]
        async with rate_limiter:
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
    """Retrieve a single Gmail message by ID."""
    gmail = await get_gmail_service(aiogoogle)
    params: Dict[str, Any] = {"userId": "me", "id": msg_id, "format": format}
    if format == "metadata" and metadata_headers:
        params["metadataHeaders"] = metadata_headers

    async with rate_limiter:
        message = await aiogoogle.as_user(gmail.users.messages.get(**params))
    return message


async def get_attachment(aiogoogle: Aiogoogle, msg_id: str, attachment_id: str) -> bytes:
    """Download an attachment from Gmail."""
    gmail = await get_gmail_service(aiogoogle)
    async with rate_limiter:
        attachment = await aiogoogle.as_user(
            gmail.users.messages.attachments.get(userId="me", messageId=msg_id, id=attachment_id)
        )
    data = attachment.get("data", "")
    return decode_base64url(data)


def get_message_attachments(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract metadata for attachments from a Gmail message."""
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
    Ensure a Gmail label exists, or create it if not found.
    Returns the label's ID.
    """
    if label_name not in ALL_LABELS and label_name != PARENT_LABEL:
        raise ValueError(f"Invalid label name: {label_name}")

    # First try to find existing label
    existing_id = await find_label_by_name(aiogoogle, label_name)
    if existing_id:
        return existing_id

    # Create parent label if needed
    if label_name.startswith(f"{PARENT_LABEL}/"):
        await ensure_parent_label_exists(aiogoogle)

    # Create the requested label
    return await create_label(aiogoogle, label_name)


async def find_label_by_name(aiogoogle: Aiogoogle, label_name: str) -> Optional[str]:
    """Find a Gmail label by name and return its ID if found."""
    gmail = await get_gmail_service(aiogoogle)
    async with rate_limiter:
        response = await aiogoogle.as_user(gmail.users.labels.list(userId="me"))

    for label in response.get("labels", []):
        if label.get("name", "").lower() == label_name.lower():
            logger.debug("Found existing label '%s' with ID: %s", label_name, label.get("id"))
            return label.get("id")
    return None


async def ensure_parent_label_exists(aiogoogle: Aiogoogle) -> None:
    """Ensure the parent label exists, creating it if needed."""
    parent_id = await find_label_by_name(aiogoogle, PARENT_LABEL)
    if not parent_id:
        logger.info("Creating parent label: %s", PARENT_LABEL)
        try:
            await create_label(aiogoogle, PARENT_LABEL)
        except Exception as e:
            if "Label name exists or conflicts" not in str(e):
                raise


async def create_label(aiogoogle: Aiogoogle, label_name: str) -> str:
    """Create a new Gmail label and return its ID."""
    gmail = await get_gmail_service(aiogoogle)
    logger.info("Creating new Gmail label: %s", label_name)

    try:
        async with rate_limiter:
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
            existing_id = await find_label_by_name(aiogoogle, label_name)
            if existing_id:
                return existing_id
        raise


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1.0),
    reraise=True,
)
async def update_message_labels(
    aiogoogle: Aiogoogle,
    msg_id: str,
    label_ids: List[str],
    label_names: List[str],
) -> bool:
    """Add label IDs to a Gmail message, retrying on failure."""
    if not label_ids:
        logger.warning("No label IDs provided for message %s", msg_id)
        return False

    gmail = await get_gmail_service(aiogoogle)
    async with rate_limiter:
        response = await aiogoogle.as_user(
            gmail.users.messages.modify(
                userId="me",
                id=msg_id,
                json={"addLabelIds": label_ids, "removeLabelIds": []},
            )
        )

    if response and "labelIds" in response:
        applied_labels = set(response["labelIds"])
        if set(label_ids).issubset(applied_labels):
            logger.debug(
                "Labels applied to message %s:\nRequested: %s\nCurrent: %s",
                msg_id,
                label_names,
                response["labelIds"],
            )
            return True
        missing = set(label_ids) - applied_labels
        logger.warning(
            "Some requested labels missing on message %s: %s\nCurrent: %s",
            msg_id,
            list(missing),
            response["labelIds"],
        )
        return False
    logger.warning("Unexpected response for message %s: %s", msg_id, response)
    return False


async def remove_empty_labels(aiogoogle: Aiogoogle) -> None:
    """Remove any empty zola child labels; ensure system labels exist."""
    gmail = await get_gmail_service(aiogoogle)
    logger.info("Ensuring system-level labels exist")
    await asyncio.gather(
        get_or_create_label(aiogoogle, PARENT_LABEL),
        get_or_create_label(aiogoogle, LLM_PARENT_LABEL),
    )
    async with rate_limiter:
        response = await aiogoogle.as_user(gmail.users.labels.list(userId="me"))

    preserved = {PARENT_LABEL.lower(), LLM_PARENT_LABEL.lower()}
    for label in response.get("labels", []):
        name = label.get("name", "")
        if name.lower().startswith(f"{PARENT_LABEL.lower()}/") and name.lower() not in preserved:
            label_id = label.get("id")
            async with rate_limiter:
                messages_response = await aiogoogle.as_user(
                    gmail.users.messages.list(userId="me", labelIds=[label_id], maxResults=1)
                )
            if not messages_response.get("messages"):
                logger.info("Removing empty label: %s", name)
                try:
                    async with rate_limiter:
                        await aiogoogle.as_user(gmail.users.labels.delete(userId="me", id=label_id))
                except Exception as e:
                    logger.warning("Failed to delete label %s: %s", name, e)


async def get_email_body_markdown(
    aiogoogle: Aiogoogle,
    message: Dict[str, Any],
    include_attachment_content: bool = True,
) -> str:
    """Extract the email body, convert to Markdown, and optionally parse attachments."""
    content = _extract_main_content(message)
    content = await _convert_html_to_markdown(content, message)

    if not include_attachment_content:
        return content

    attachments = get_message_attachments(message)
    if not attachments:
        return content

    attachment_content = await _process_attachments(aiogoogle, message, attachments)
    if not attachment_content:
        return content

    return f"{content}\n\n{attachment_content}"


def _extract_main_content(message: Dict[str, Any]) -> str:
    """Extract the main content from the message."""
    payload = message.get("payload", {})
    if not payload:
        return message.get("snippet", "").strip()

    content = extract_text_content(payload)
    if not content:
        return message.get("snippet", "").strip()

    return content


async def _convert_html_to_markdown(content: str, message: Dict[str, Any]) -> str:
    """Convert HTML content to Markdown if needed using async file I/O."""
    payload = message.get("payload", {})
    if not payload.get("mimeType", "").startswith("text/html"):
        return content

    md = MarkItDown()
    # Use the asynchronous temporary file manager and aiofiles for non-blocking I/O
    import aiofiles  # Import here if not already imported at the top

    async with async_temporary_file(".html") as tmp_path:
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            await f.write(content)
        # Offload the blocking conversion to a thread so as not to block the async loop.
        result = await asyncio.to_thread(md.convert, str(tmp_path))

    if result and result.text_content:
        return result.text_content.strip()
    return content


async def _process_attachments(
    aiogoogle: Aiogoogle,
    message: Dict[str, Any],
    attachments: List[Dict[str, Any]],
) -> Optional[str]:
    """Process message attachments and return formatted content."""
    if not attachments:
        return None

    content_parts = ["## Attachments"]

    for att in attachments:
        attachment_info = _format_attachment_info(att)
        content_parts.append(attachment_info)

        attachment_content = await _get_attachment_content(aiogoogle, message, att)
        if attachment_content:
            content_parts.append(attachment_content)

    return "\n\n".join(content_parts)


def _format_attachment_info(attachment: Dict[str, Any]) -> str:
    """Format basic attachment information."""
    filename = attachment.get("filename")
    mime = attachment.get("mimeType")
    size = attachment.get("size")
    return f"- **{filename}** ({mime}, {size} bytes)"


async def _get_attachment_content(
    aiogoogle: Aiogoogle,
    message: Dict[str, Any],
    attachment: Dict[str, Any],
) -> Optional[str]:
    """Extract and convert attachment content if supported."""
    supported_mime = {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    mime = attachment.get("mimeType")
    if not (mime.startswith("text/") or mime in supported_mime):
        return None

    try:
        data = await _get_attachment_data(aiogoogle, message, attachment)
        if not data:
            return None

        return await _convert_attachment_to_markdown(data, attachment.get("filename", ""))
    except Exception as e:
        logger.warning("Failed to process attachment %s (%s): %s", attachment.get("filename"), mime, e)
        return None


async def _get_attachment_data(
    aiogoogle: Aiogoogle,
    message: Dict[str, Any],
    attachment: Dict[str, Any],
) -> Optional[bytes]:
    """Get attachment data either from payload or by downloading."""
    filename = attachment.get("filename")
    payload = message.get("payload", {})

    # Check inline data first
    for part in payload.get("parts", []):
        if part.get("filename") == filename and part.get("body", {}).get("data"):
            return decode_base64url(part["body"]["data"])

    # Download if not found inline
    if attachment.get("attachmentId"):
        return await get_attachment(aiogoogle, message["id"], attachment["attachmentId"])

    return None


async def _convert_attachment_to_markdown(data: bytes, filename: str) -> Optional[str]:
    """Convert attachment data to markdown format."""
    with temporary_file(Path(filename).suffix or ".bin") as tmp_path:
        with open(tmp_path, "wb") as f:
            f.write(data)

        md = MarkItDown()
        result = md.convert(str(tmp_path))
        if result and result.text_content.strip():
            return f"\n```\n{result.text_content.strip()}\n```"

    return None
