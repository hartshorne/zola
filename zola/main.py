import argparse
import asyncio
import logging
import signal
import sys
import traceback
import tracemalloc
from pathlib import Path
from typing import Any, Optional, Tuple

from rich.logging import RichHandler
from rich.panel import Panel

from zola.account_manager import AccountManager
from zola.console import console
from zola.gmail_utils import remove_empty_labels
from zola.llm_utils import SafeTokenizer, classify_email, load_llm
from zola.processing import SLAData, process_inbox_folder, process_sent_folder

# Constants
CREDENTIALS_PATH = Path("secrets/credentials/application_credentials.json")
SUPPORTED_MODELS = [
    "mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit",
    "mlx-community/Llama-3.3-70B-Instruct-8bit",
    "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
    "mlx-community/Mistral-Small-24B-Instruct-2501-8bit",
]
DEFAULT_MODEL = "mlx-community/Mistral-Small-24B-Instruct-2501-4bit"

tracemalloc.start(10)  # keep 10 frames of traceback
logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """
    Configure the logging system based on the specified log level.

    Args:
        log_level: String representing the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Raises:
        ValueError: If an invalid log level is specified
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logging.getLogger().setLevel(numeric_level)


def check_credentials() -> None:
    """
    Verify that OAuth credentials exist at the expected location.

    Raises:
        FileNotFoundError: If credentials file is not found
    """
    if not CREDENTIALS_PATH.exists():
        console.print(
            Panel.fit(
                (
                    f"[bold red]Error[/bold red]: OAuth credentials not found at [bold]{CREDENTIALS_PATH}[/bold]\n\n"
                    "See the README.md for instructions on setting up credentials."
                ),
                title="[bold red]Credentials Missing[/bold red]",
                border_style="red",
            )
        )
        raise FileNotFoundError(f"Credentials not found at {CREDENTIALS_PATH}")


async def test_llm(model_identifier: str) -> Tuple[Any, SafeTokenizer]:
    console.print(Panel("Testing LLM installation...", style="bold magenta", expand=False))
    with console.status("[bold magenta]Loading LLM model and tokenizer..."):
        model, tokenizer = load_llm(model_identifier)

    test_context = """
User Context:
  User Name: Test User
  User Email: test@example.com

Email Metadata:
  Subject: Quick test email
  To: test@example.com
  From: sender@example.com
  Typical SLA for sender domain (example.com): N/A

Email Body (Markdown):
This is a test email to verify the LLM is working.
    """.strip()

    console.print("[bold magenta]Running test classification...[/bold magenta]")
    labels = await classify_email(test_context, model, tokenizer)

    if "error" in labels:
        console.print(
            Panel.fit(
                "[bold red]LLM test failed![/bold red]\n\nThe LLM returned an error in its classification.",
                title="[bold red]LLM Test Failed[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)

    console.print(
        Panel.fit(
            f"[bold green]LLM test successful![/bold green]\n\nTest classification: [bold cyan]{labels}[/bold cyan]",
            title="[bold green]LLM Test Complete[/bold green]",
            border_style="green",
        )
    )
    return model, tokenizer


async def reauthorize_account(account_mgr: AccountManager, account_name: str) -> bool:
    """
    Attempt to reauthorize an account and handle errors.

    Args:
        account_mgr: The account manager instance
        account_name: Name of the account to reauthorize

    Returns:
        bool: True if reauthorization succeeded, False otherwise
    """
    try:
        console.print(f"[blue]Let's reconnect your [bold]{account_name}[/bold] account...[/blue]")
        await account_mgr.reauthorize_account(account_name)
        console.print(
            Panel.fit(
                f"[green]✓ Successfully reconnected to [bold]{account_name}[/bold]![/green]",
                border_style="green",
            )
        )
        return True
    except Exception as reauth_err:
        if "User canceled" in str(reauth_err):
            console.print(f"[yellow]Skipping account '{account_name}'[/yellow]")
        else:
            logger.debug(f"Reauthorization failed: {reauth_err}")
            console.print(
                Panel.fit(
                    f"[orange3]I couldn't reconnect to '{account_name}':[/orange3]\n\n"
                    f"[bold]Error:[/bold] {str(reauth_err).strip()}\n\n"
                    f"[dim]Please try again later or check your network connection.[/dim]",
                    title="[orange3]⚠️ Reconnection Failed[/orange3]",
                    border_style="orange3",
                )
            )
        return False


async def handle_auth_error(account_mgr: AccountManager, account_name: str, user_email: str) -> bool:
    """
    Handle authentication errors by attempting to reauthorize the account.

    Args:
        account_mgr: The account manager instance
        account_name: Name of the account with auth error
        user_email: Email address associated with the account

    Returns:
        bool: True if reauthorization succeeded, False otherwise
    """
    console.print(
        Panel.fit(
            f"[orange3]Your '{account_name}' account needs to be reconnected[/orange3]\n\n"
            f"Gmail authentication has expired for [bold]{user_email}[/bold].\n"
            f"This happens periodically and is normal.",
            title="[orange3]⚠️ Authentication Expired[/orange3]",
            border_style="orange3",
        )
    )

    return await reauthorize_account(account_mgr, account_name)


async def get_authorized_service(account_mgr: AccountManager, account_name: str, user_email: str):
    """
    Get an authorized service for the account, handling auth errors.

    Args:
        account_mgr: The account manager instance
        account_name: Name of the account to get service for
        user_email: Email address associated with the account

    Returns:
        The aiogoogle service or None if authorization failed
    """
    try:
        return await account_mgr.get_service(account_name)
    except Exception as e:
        error_str = str(e).strip()
        if "invalid_grant" in error_str:
            if await handle_auth_error(account_mgr, account_name, user_email):
                return await account_mgr.get_service(account_name)
            return None
        else:
            # For other errors, show generic error message
            logger.debug(f"Error setting up account {account_name}: {str(e)}")
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            console.print(
                Panel.fit(
                    f"[orange3]I couldn't set up your '{account_name}' account:[/orange3]\n\n"
                    f"[bold]Error:[/bold] {error_str}\n\n"
                    f"[dim]Try restarting Zola or checking your internet connection.[/dim]",
                    title="[orange3]⚠️ Account Setup Issue[/orange3]",
                    border_style="orange3",
                )
            )
            return None


async def process_gmail_account(
    aiogoogle, llm_model: Any, tokenizer: SafeTokenizer, max_messages: int, account_name: str
) -> bool:
    """
    Process Gmail account data once authentication is successful.

    Args:
        aiogoogle: The authenticated Google service
        llm_model: The loaded LLM model
        tokenizer: The tokenizer for the LLM model
        max_messages: Maximum number of messages to process
        account_name: Name of the account being processed

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        console.print("[dim]Organizing your labels...[/dim]")
        await remove_empty_labels(aiogoogle)

        sla_data = SLAData()
        await process_sent_folder(aiogoogle, sla_data, max_messages)
        await process_inbox_folder(
            aiogoogle=aiogoogle,
            llm_model=llm_model,
            tokenizer=tokenizer,
            sla_data=sla_data,
            max_messages=max_messages,
        )

        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            sla_data.print_debug_stats()

        return True
    except Exception as e:
        error_message = str(e).strip()
        logger.debug(f"Error processing account {account_name}: {str(e)}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")

        # Generic error handling for non-auth errors
        if "invalid_grant" not in error_message:
            help_message = "[dim]You may need to check your internet connection or try again later.[/dim]"
            console.print(
                Panel.fit(
                    f"[orange3]I had trouble processing your '{account_name}' account:[/orange3]\n\n"
                    f"[bold]Error:[/bold] {error_message}\n\n"
                    f"{help_message}",
                    title="[orange3]⚠️ Account Issue[/orange3]",
                    border_style="orange3",
                )
            )
        return False


async def process_account(
    account_mgr: AccountManager,
    account_name: str,
    llm_model: Any,
    tokenizer: SafeTokenizer,
    max_messages: int,
) -> None:
    """
    Process a single Gmail account using the provided LLM model and tokenizer.
    """
    try:
        user_info = account_mgr.get_account(account_name)
        console.print(
            Panel(
                f"[bold blue]👤 Working with:[/bold blue] [bold]{user_info.shortname} ({user_info.email})[/bold]",
                title="[bold blue]Processing Account[/bold blue]",
                border_style="blue",
            )
        )

        # Get authorized service
        aiogoogle = await get_authorized_service(account_mgr, account_name, user_info.email)
        if not aiogoogle:
            return  # Skip this account if we couldn't authenticate

        async with aiogoogle:
            success = await process_gmail_account(aiogoogle, llm_model, tokenizer, max_messages, account_name)

            # If processing failed due to auth error, try to reconnect and process again
            if not success:
                error_message = "invalid_grant"  # Assumed from the control flow

                if "invalid_grant" in error_message:
                    if await handle_auth_error(account_mgr, account_name, user_info.email):
                        # Try processing again with new auth
                        aiogoogle = await account_mgr.get_service(account_name)
                        if aiogoogle:
                            async with aiogoogle:
                                await process_gmail_account(aiogoogle, llm_model, tokenizer, max_messages, account_name)

    except Exception as e:
        # Handle unexpected errors not caught by more specific handlers
        logger.debug(f"Unexpected error with account {account_name}: {str(e)}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        console.print(
            Panel.fit(
                f"[orange3]Unexpected error with '{account_name}':[/orange3]\n\n"
                f"[bold]Error:[/bold] {str(e).strip()}\n\n"
                f"[dim]Please report this issue if it persists.[/dim]",
                title="[orange3]⚠️ Unexpected Error[/orange3]",
                border_style="orange3",
            )
        )


async def async_main(
    max_messages: int,
    model_identifier: str,
    account_name: Optional[str] = None,
    skip_llm_test: bool = False,
) -> None:
    """
    Main async entry point for the application.

    Args:
        max_messages: Maximum number of messages to process per account
        model_identifier: Name/path of the LLM model to use
        account_name: Specific account to process (None to process all or select interactively)
        skip_llm_test: Whether to skip the initial LLM functionality test
    """
    # Set up signal handlers for clean shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(shutdown(sig)))

    # Check credentials
    check_credentials()

    # Load model
    if not skip_llm_test:
        llm_model, tokenizer = await test_llm(model_identifier)
    else:
        llm_model, tokenizer = load_llm(model_identifier)

    # Set up accounts
    account_mgr = AccountManager()
    # Wait for account validation to complete before proceeding
    await account_mgr.validate_all_accounts()

    if account_name is None:
        account_names = account_mgr.get_account_names()
        if not account_names:
            account_name, _ = await account_mgr.select_account()
            account_names = [account_name]
    else:
        account_names = [account_name]

    # Process accounts concurrently using gather with structured task creation
    tasks = []
    for account in account_names:
        tasks.append(process_account(account_mgr, account, llm_model, tokenizer, max_messages))

    await asyncio.gather(*tasks)

    # Show completion message
    console.print(
        Panel.fit(
            "✨ [bold green]Email organization complete![/bold green] ✨\n\n"
            f"Processed {len(account_names)} account{'s' if len(account_names) != 1 else ''} with up to {max_messages} messages each.\n"
            "[dim]Your emails should now be neatly organized with Zola's labels.[/dim]",
            title="[bold green]Success![/bold green]",
            border_style="green",
        )
    )


async def shutdown(signal: signal.Signals) -> None:
    """
    Handle graceful shutdown on signals.

    Args:
        signal: The signal that triggered the shutdown
    """
    logger.info(f"Received exit signal {signal.name}")

    # Get all tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    if tasks:
        logger.info(f"Cancelling {len(tasks)} outstanding tasks")
        # Cancel all tasks
        for task in tasks:
            task.cancel()

        # Wait for all tasks to complete with cancellation
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Shutdown complete")


def main() -> None:
    """
    Main entry point for the application. Parses command-line arguments and starts the async workflow.
    """
    parser = argparse.ArgumentParser(description="Zola: A Gmail classifier using an LLM.")
    parser.add_argument(
        "--max-messages",
        "-m",
        type=int,
        default=10,
        help="Maximum number of messages to process (default: 10).",
    )
    parser.add_argument(
        "--model",
        "-M",
        type=str,
        default=DEFAULT_MODEL,
        choices=SUPPORTED_MODELS,
        help="LLM model identifier/path to use",
    )
    parser.add_argument(
        "--account",
        "-a",
        type=str,
        help="Account name to use (if not specified, will prompt for selection)",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--skip-llm-test",
        "-s",
        action="store_true",
        help="Skip the LLM test at startup",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Enable tracemalloc for debugging memory leaks
    tracemalloc.start()

    # Use asyncio.run which properly manages the event loop lifecycle
    try:
        asyncio.run(
            async_main(
                max_messages=args.max_messages,
                model_identifier=args.model,
                account_name=args.account,
                skip_llm_test=args.skip_llm_test,
            )
        )
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        console.print("[yellow]Operation cancelled by user. Shutting down...[/yellow]")
    finally:
        # Stop tracemalloc
        tracemalloc.stop()


if __name__ == "__main__":
    main()
