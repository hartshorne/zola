import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

from rich.logging import RichHandler
from rich.panel import Panel

from zola.account_manager import AccountManager
from zola.console import console
from zola.gmail_utils import remove_empty_labels
from zola.llm_utils import classify_email, load_llm
from zola.processing import SLAData, process_inbox_folder, process_sent_folder


def setup_logging(log_level: str) -> None:
    """
    Configure logging with the specified level, using Rich for colored logging output.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",  # RichHandler handles formatting
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    logging.getLogger().setLevel(numeric_level)


def check_credentials() -> None:
    """
    Verify that the application credentials exist in the expected location.
    """
    creds_path = Path("secrets/credentials/application_credentials.json")
    if not creds_path.exists():
        console.print(
            Panel.fit(
                (
                    f"[bold red]Error[/bold red]: OAuth credentials not found at [bold]{creds_path}[/bold]\n\n"
                    "See the README.md for detailed instructions on setting up credentials."
                ),
                title="[bold red]Credentials Missing[/bold red]",
                border_style="red",
            )
        )
        raise FileNotFoundError(f"Credentials not found at {creds_path}")


async def test_llm(model_identifier: str) -> Tuple:
    """
    Run a quick test of the LLM to ensure it's working.
    Returns the loaded model and tokenizer if the test passes.
    """
    console.print(Panel("Testing LLM installation...", style="bold magenta", expand=False))
    with console.status("[bold magenta]Loading LLM model and tokenizer...") as status:
        model, tokenizer = load_llm(model_identifier)
        status.update("[bold magenta]Model loaded successfully. Preparing test classification...")

        test_context = (
            """
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
            """
        ).strip()

        status.update("[bold magenta]Running test classification...")
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


async def async_main(
    max_messages: int,
    model_identifier: str,
    account_name: Optional[str] = None,
    skip_llm_test: bool = False,
) -> None:
    # Verify that the credentials file exists.
    check_credentials()

    # Load and (optionally) test the LLM.
    if not skip_llm_test:
        llm_model, tokenizer = await test_llm(model_identifier)
    else:
        llm_model, tokenizer = load_llm(model_identifier)

    account_mgr = AccountManager()

    # Get list of accounts to process
    if account_name is None:
        account_names = account_mgr.get_account_names()
        if not account_names:
            # If no accounts exist, prompt to create one
            account_name, aiogoogle = await account_mgr.select_account()
            account_names = [account_name]
    else:
        account_names = [account_name]

    # Process each account
    for account_name in account_names:
        console.print(
            Panel(
                f"[bold blue]Processing account:[/bold blue] [bold]{account_name}[/bold]",
                title="Account Selection",
                border_style="blue",
            )
        )

        aiogoogle = await account_mgr.get_service(account_name)
        async with aiogoogle:  # Ensure proper session management
            console.print("[bold blue]Checking for empty labels...[/bold blue]")
            await remove_empty_labels(aiogoogle)

            # Process the sent messages to build SLA data.
            sla_data = SLAData()
            await process_sent_folder(aiogoogle, sla_data, max_messages)

            # Process the inbox messages
            await process_inbox_folder(
                aiogoogle=aiogoogle,
                llm_model=llm_model,
                tokenizer=tokenizer,
                sla_data=sla_data,
                max_messages=max_messages,
            )

            # If in debug mode, print additional SLA statistics.
            if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                sla_data.print_debug_stats()

    console.print(Panel("[bold green]All done![/bold green]", border_style="green"))


def main() -> None:
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
        default="mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
        choices=[
            "mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit",
            "mlx-community/Llama-3.3-70B-Instruct-8bit",
            "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
            "mlx-community/Mistral-Small-24B-Instruct-2501-8bit",
        ],
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

    asyncio.run(
        async_main(
            max_messages=args.max_messages,
            model_identifier=args.model,
            account_name=args.account,
            skip_llm_test=args.skip_llm_test,
        )
    )


if __name__ == "__main__":
    main()
