#!/usr/bin/env python3
"""
test_zola.py

Unit tests and doctests for the Zola package.
Run these tests with:
    python -m unittest discover -s tests
"""

import asyncio
import base64
import os
import time
import unittest
import zoneinfo
from datetime import datetime, timezone
from typing import Dict, List
from unittest.mock import AsyncMock, patch

from zola.gmail_utils import (
    convert_to_user_timezone,
    extract_email_domain,
    format_datetime_for_model,
    get_email_body_markdown,
    get_user_timezone,
    parse_date,
)
from zola.llm_utils import classify_email
from zola.processing import process_inbox_message
from zola.settings import LLM_PARENT_LABEL, PARENT_LABEL


class TestGmailUtils(unittest.TestCase):
    def test_parse_date(self):
        test_cases = [
            (
                "Fri, 31 Jan 2025 22:14:17 -0800",
                {"year": 2025, "month": 1, "day": 31, "hour": 22, "minute": 14, "second": 17},
            ),
            ("31 Jan 2025 22:14:17 -0800", {"year": 2025, "month": 1, "day": 31}),
            ("Fri, 31 Jan 2025 22:14:17 GMT", {"year": 2025, "month": 1, "day": 31}),
            ("Fri, 31 Jan 2025 22:14:17 +0000", {"year": 2025, "month": 1, "day": 31}),
            (
                "Mon, 1 Jan 2025 00:00:00 +0000",
                {"year": 2025, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0},
            ),
            ("Thu, 29 Feb 2024 12:00:00 +0000", {"year": 2024, "month": 2, "day": 29}),
        ]
        for date_str, expected in test_cases:
            with self.subTest(date_str=date_str):
                dt = parse_date(date_str)
                for attr, value in expected.items():
                    self.assertEqual(getattr(dt, attr), value)
                self.assertIsNotNone(dt.tzinfo)
        invalid_dates = ["", "Not a date", "2025-01-31", "31/01/2025", "Fri, 31 Foo 2025 22:14:17 -0800"]
        for date_str in invalid_dates:
            with self.subTest(date_str=date_str):
                with self.assertRaises(Exception):
                    parse_date(date_str)

    def test_timezone_handling(self):
        date1 = parse_date("Fri, 31 Jan 2025 22:14:17 +0000")
        date2 = parse_date("Fri, 31 Jan 2025 14:14:17 -0800")
        date3 = parse_date("Fri, 31 Jan 2025 22:14:17")
        self.assertEqual(date1, date2)
        self.assertEqual(date1, date3)
        self.assertEqual((date2 - date1).total_seconds(), 0)
        self.assertEqual((date3 - date1).total_seconds(), 0)
        test_cases = [("America/New_York", "America/New_York"), ("Invalid/Timezone", None), ("", None)]
        for tz_input, expected in test_cases:
            with self.subTest(tz=tz_input):
                with patch.dict(os.environ, {"TZ": tz_input}):
                    user_tz = get_user_timezone()
                    if expected is not None:
                        self.assertEqual(user_tz, expected)
                    else:
                        self.assertIn(user_tz, zoneinfo.available_timezones())
        with patch.dict(os.environ, {"TZ": "America/Los_Angeles"}):
            utc_dt = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
            local_dt = convert_to_user_timezone(utc_dt)
            self.assertIn(local_dt.hour, (4, 5))
            self.assertEqual(getattr(local_dt.tzinfo, "key", None), "America/Los_Angeles")

    def test_extract_email_domain(self):
        test_cases = [
            ("john@example.com", "example.com"),
            ("John Doe <john@example.com>", "example.com"),
            ("<john@example.com>", "example.com"),
            ("John <john@example.com>, Jane <jane@other.com>", "example.com"),
            ("user@sub.example.com", "sub.example.com"),
            ("user@sub.sub2.example.co.uk", "sub.sub2.example.co.uk"),
            ("user@EXAMPLE.COM", "example.com"),
            ("user+label@example.com", "example.com"),
            ("user.name@example.com", "example.com"),
            ("", "unknown"),
            ("No email here", "unknown"),
            ("@example.com", "unknown"),
            ("user@", "unknown"),
            ("Multiple @ symbols@not@valid.com", "unknown"),
        ]
        for input_str, expected_domain in test_cases:
            with self.subTest(input_str=input_str):
                self.assertEqual(extract_email_domain(input_str), expected_domain)

    def test_datetime_formatting(self):
        with patch.dict(os.environ, {"TZ": "America/Los_Angeles"}):
            test_cases = [
                (
                    datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
                    lambda f: "2025-01-01" in f and ("PST" in f or "PDT" in f),
                ),
                (datetime(2024, 3, 10, 10, 0, tzinfo=timezone.utc), lambda f: "PDT" in f),
                (datetime(2024, 11, 3, 9, 0, tzinfo=timezone.utc), lambda f: "PST" in f),
            ]
            for dt, validator in test_cases:
                with self.subTest(dt=dt):
                    formatted = format_datetime_for_model(dt)
                    self.assertTrue(validator(formatted))


class TestLLMUtils(unittest.IsolatedAsyncioTestCase):
    async def test_classify_email(self):
        import logging

        logging.disable(logging.CRITICAL)
        try:

            class MockTokenizer:
                def __init__(self):
                    self.model_max_length = 8192
                    self.vocab = {"test": 1}
                    self.eos_token_id = 0

                def encode(self, text):
                    return [1, 2, 3]

                def decode(self, tokens, skip_special_tokens=True):
                    return "test"

            test_cases = [
                ('["priority", "respond"]', ["priority", "respond"]),
                ("not a valid python list", [f"{LLM_PARENT_LABEL}/confused"]),
                ('[123, "test", true]', [f"{LLM_PARENT_LABEL}/confused"]),
                ("[]", [f"{LLM_PARENT_LABEL}/confused"]),
                ('["PRIORITY", "RESPOND"]', ["priority", "respond"]),
                (
                    """Based on the email content, here are the relevant labels:
                ["priority", "respond"]""",
                    ["priority", "respond"],
                ),
                (f'["{PARENT_LABEL}/priority"]', ["priority"]),
                (f'["{LLM_PARENT_LABEL}/confused"]', [f"{LLM_PARENT_LABEL}/confused"]),
            ]
            for mock_output, expected in test_cases:
                with self.subTest(mock_output=mock_output):
                    with patch("zola.llm_utils.mlx_generate", return_value=mock_output):
                        result = await classify_email("test context", None, MockTokenizer())
                        self.assertEqual(result, expected)
                        for label in result:
                            if label.startswith(f"{PARENT_LABEL}/{PARENT_LABEL}"):
                                self.fail(f"Label {label} has double prefix")
        finally:
            logging.disable(logging.NOTSET)


class TestGmailRateLimiting(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limiter(self):
        from zola.gmail_utils import RateLimiter

        limiter = RateLimiter(max_calls=10, period=1.0)
        start_time = time.monotonic()
        for _ in range(5):
            async with limiter:
                pass
        elapsed = time.monotonic() - start_time
        self.assertGreaterEqual(elapsed, 0.4, "Rate limiter didn't properly space out requests")
        start_time = time.monotonic()

        async def make_request():
            async with limiter:
                return True

        tasks = [make_request() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start_time
        self.assertGreaterEqual(elapsed, 1.0, "Rate limiter didn't properly handle concurrent requests")
        self.assertEqual(len(results), 20, "Not all requests completed")
        self.assertTrue(all(results), "Some requests failed")


class TestEmailBodyProcessing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.maxDiff = None

    def create_mock_message(self, mime_type: str, content: str, attachments: List[Dict] = None) -> Dict:
        encoded_content = base64.urlsafe_b64encode(content.encode()).decode()
        message = {
            "id": "test_message_id",
            "threadId": "test_thread_id",
            "labelIds": ["INBOX"],
            "snippet": "Email snippet",
            "payload": {
                "partId": "",
                "mimeType": mime_type,
                "filename": "",
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "Subject", "value": "Test Email"},
                    {"name": "Content-Type", "value": f"{mime_type}; charset=UTF-8"},
                ],
            },
        }
        if mime_type == "multipart/mixed":
            message["payload"]["parts"] = []
            if content:
                message["payload"]["parts"].append(
                    {
                        "partId": "0",
                        "mimeType": "text/plain",
                        "filename": "",
                        "headers": [{"name": "Content-Type", "value": "text/plain; charset=UTF-8"}],
                        "body": {"size": len(content), "data": encoded_content},
                    }
                )
            if attachments:
                for i, attachment in enumerate(attachments, 1):
                    part = {
                        "partId": str(i),
                        "mimeType": attachment["type"],
                        "filename": attachment["name"],
                        "headers": [
                            {"name": "Content-Type", "value": f'{attachment["type"]}; name="{attachment["name"]}"'},
                            {"name": "Content-Disposition", "value": f'attachment; filename="{attachment["name"]}"'},
                        ],
                        "body": {"attachmentId": attachment.get("id", "test_id"), "size": attachment.get("size", 0)},
                    }
                    if "content" in attachment:
                        part["body"]["data"] = base64.urlsafe_b64encode(attachment["content"].encode()).decode()
                    message["payload"]["parts"].append(part)
        else:
            message["payload"]["body"] = {"size": len(content), "data": encoded_content}
        return message

    async def test_email_content_processing(self):
        # Patch get_attachment to avoid PDF conversion errors.
        # Return bytes instead of str to avoid conversion errors.
        with patch("zola.gmail_utils.get_attachment", new=AsyncMock(return_value=b"Dummy Attachment Content")):
            # Plain text snippet.
            message = {"snippet": "Test snippet"}
            result = await get_email_body_markdown(None, message)
            self.assertEqual(result.strip(), "Test snippet")
            # Message part.
            text = "Hello World"
            encoded = base64.urlsafe_b64encode(text.encode("ASCII")).decode("ASCII")
            message = {
                "payload": {"parts": [{"mimeType": "text/plain", "body": {"data": encoded}}]},
                "snippet": text,
            }
            result = await get_email_body_markdown(None, message)
            self.assertEqual(result.strip(), text.strip())
            # Empty message.
            self.assertEqual((await get_email_body_markdown(None, {})).strip(), "")
            # Various formats.
            test_cases = [
                ("text/plain", "This is a plain text email.\nWith multiple lines.\n", None),
                (
                    "text/html",
                    "<html><body><h1>Hello</h1><p>This is an HTML email.</p></body></html>",
                    None,
                    "# Hello\n\nThis is an HTML email.",
                ),
                (
                    "multipart/mixed",
                    "Main email content",
                    [
                        {"type": "application/pdf", "name": "test.pdf", "id": "pdf123", "size": 1024},
                        {"type": "text/plain", "name": "notes.txt", "content": "Text file content", "id": "txt123"},
                    ],
                ),
                ("text/plain", "", None),
            ]

            # Dummy aiogoogle client for attachments.
            class DummyLabels:
                async def list(self, userId):
                    return {"labels": []}

            class DummyAttachments:
                async def get(self, userId, messageId, id):
                    return {"data": base64.urlsafe_b64encode(b"Attachment content").decode()}

            class DummyMessages:
                attachments = DummyAttachments()

            class DummyUsers:
                messages = DummyMessages()
                labels = DummyLabels()

            class MockAiogoogle:
                users = DummyUsers()

                async def discover(self, *args, **kwargs):
                    return self

                async def as_user(self, *args, **kwargs):
                    return {"data": base64.urlsafe_b64encode(b"Attachment content").decode()}

            mock_aiogoogle = MockAiogoogle()
            for test_case in test_cases:
                with self.subTest(mime_type=test_case[0]):
                    if len(test_case) == 3:
                        mime_type, content, attachments = test_case
                        expected_content = content.strip()
                    else:
                        mime_type, content, attachments, expected_content = test_case
                    message = self.create_mock_message(mime_type, content, attachments)
                    # For multipart/mixed messages, patch the PDF converter to avoid errors.
                    if mime_type == "multipart/mixed":
                        with patch("markitdown._markitdown.MarkItDown.convert_local", return_value="Dummy PDF Content"):
                            result = await get_email_body_markdown(
                                mock_aiogoogle,
                                message,
                                include_attachment_content=bool(attachments),
                            )
                    else:
                        result = await get_email_body_markdown(
                            mock_aiogoogle,
                            message,
                            include_attachment_content=bool(attachments),
                        )
                    self.assertIsInstance(result, str)
                    if content:
                        self.assertIn(expected_content, result)
                    if attachments:
                        self.assertIn("## Attachments", result)
                        for attachment in attachments:
                            self.assertIn(attachment["name"], result)

    def test_attachment_metadata(self):
        from zola.gmail_utils import get_message_attachments

        attachments = [
            {"type": "application/pdf", "name": "document.pdf", "size": 1024, "id": "pdf123"},
            {"type": "image/jpeg", "name": "photo.jpg", "size": 2048, "id": "jpg123"},
        ]
        message = self.create_mock_message("multipart/mixed", "content", attachments)
        extracted = get_message_attachments(message)
        self.assertEqual(len(extracted), 2)
        for i, expected in enumerate(attachments):
            attachment = extracted[i]
            self.assertEqual(attachment["filename"], expected["name"])
            self.assertEqual(attachment["mimeType"], expected["type"])
            self.assertEqual(attachment["size"], expected["size"])
            self.assertEqual(attachment["attachmentId"], expected["id"])


class TestProcessing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_aiogoogle = AsyncMock()
        self.mock_gmail = AsyncMock()
        self.mock_aiogoogle.discover.return_value = self.mock_gmail
        self.mock_aiogoogle.as_user.return_value = AsyncMock()
        self.mock_profile = {
            "emailAddress": "test@example.com",
            "messagesTotal": 100,
            "threadsTotal": 100,
            "historyId": "12345",
        }
        self.mock_aiogoogle.as_user.return_value.execute = AsyncMock(return_value=self.mock_profile)
        self.mock_message = {
            "payload": {
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "Date", "value": "Thu, 1 Jan 2024 00:00:00 +0000"},
                ]
            }
        }
        self.mock_gmail.users = AsyncMock()
        self.mock_gmail.users.getProfile = AsyncMock(return_value=self.mock_profile)
        self.mock_gmail.users.messages = AsyncMock()
        self.mock_gmail.users.messages.get = AsyncMock(return_value=self.mock_message)
        self.mock_gmail.users.labels = AsyncMock()
        self.mock_gmail.users.labels.list = AsyncMock(return_value={"labels": []})

        class MockTokenizer:
            def __init__(self):
                self.model_max_length = 8192
                self.vocab = {"test": 1}
                self.eos_token_id = 0

            def encode(self, text):
                return [1, 2, 3]

            def decode(self, tokens):
                return "test"

        self.mock_tokenizer = MockTokenizer()
        self.mock_model = AsyncMock()

    @patch("zola.processing.classify_email")
    @patch("zola.processing.get_email_body_markdown")
    @patch("zola.processing.get_message")
    @patch("zola.processing.get_user_info")
    async def test_process_inbox_message_labels(
        self, mock_get_user_info, mock_get_message, mock_get_body, mock_classify_email
    ):
        mock_get_body.return_value = "Test email body"
        mock_get_message.return_value = {
            "id": "test_msg_id",
            "threadId": "test_thread_id",
            "labelIds": ["INBOX"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "Date", "value": "Thu, 1 Jan 2024 00:00:00 +0000"},
                ],
                "body": {"data": "Test body"},
            },
        }
        mock_get_user_info.return_value = ("Test User", "test@example.com")

        class MockSLAData:
            def get_sla_categories(self, sender, domain):
                return "quick", "quick"

        mock_sla_data = MockSLAData()
        test_cases = [
            (
                ["priority", "respond"],
                [f"{PARENT_LABEL}/priority", f"{PARENT_LABEL}/respond"],
            ),
            ([f"{LLM_PARENT_LABEL}/confused"], [f"{LLM_PARENT_LABEL}/confused"]),
            ([f"{PARENT_LABEL}/priority"], [f"{PARENT_LABEL}/priority"]),
            (
                [f"{PARENT_LABEL}/{LLM_PARENT_LABEL}/confused"],
                [f"{LLM_PARENT_LABEL}/confused"],
            ),
            # Test cases for school and newsletters
            (["school"], [f"{PARENT_LABEL}/school"]),
            (
                ["school", "priority"],
                [f"{PARENT_LABEL}/school", f"{PARENT_LABEL}/priority"],
            ),  # school can combine with priority
            (["newsletters"], [f"{PARENT_LABEL}/newsletters"]),
            # Test cases for ignore label
            (["ignore"], [f"{PARENT_LABEL}/ignore"]),
            (
                ["ignore", "priority"],
                [f"{PARENT_LABEL}/ignore"],
            ),  # ignore should override other labels
            (
                ["ignore", "newsletters"],
                [f"{PARENT_LABEL}/ignore"],
            ),  # ignore should override newsletters too
            (
                [f"{PARENT_LABEL}/ignore", "priority"],
                [f"{PARENT_LABEL}/ignore"],
            ),  # ignore should override even when prefixed
        ]

        for llm_output, expected_labels in test_cases:
            with self.subTest(llm_output=llm_output):
                mock_classify_email.return_value = llm_output
                result = await process_inbox_message(
                    self.mock_aiogoogle,
                    "test_msg_id",
                    mock_sla_data,
                    self.mock_model,
                    self.mock_tokenizer,
                )
                self.assertEqual(result, expected_labels)
                for label in result:
                    double_prefix = f"{PARENT_LABEL}/{PARENT_LABEL}"
                    if label.startswith(double_prefix):
                        self.fail(f"Label {label} has double prefix")
                    if f"/{PARENT_LABEL}/" in label and label.count(PARENT_LABEL) > 1:
                        self.fail(f"Label {label} has nested parent label")
                    if "ignore" in label:
                        self.assertEqual(
                            len(result),
                            1,
                            f"'ignore' label should be used alone, got: {result}",
                        )


if __name__ == "__main__":
    unittest.main()
