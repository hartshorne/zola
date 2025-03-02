import asyncio
import json
import logging
import os
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import aiofiles  # Added for async file operations
from aiogoogle.auth.creds import UserCreds
from aiogoogle.client import Aiogoogle
from rich.console import Console
from rich.panel import Panel

from zola.settings import SCOPES

logger = logging.getLogger(__name__)
console = Console()


class UserInfo:
    """Represents a Gmail account user's information."""

    def __init__(self, shortname: str, name: str, email: str):
        self.shortname = shortname
        self.name = name
        self.email = email

    @classmethod
    def from_dict(cls, shortname: str, data: Dict[str, str]) -> "UserInfo":
        """Create a UserInfo instance from account data dictionary."""
        return cls(shortname=shortname, name=data.get("name", "User"), email=data.get("email", ""))

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for storage."""
        return {"name": self.name, "email": self.email}

    def __str__(self) -> str:
        return f"{self.shortname} ({self.email} - {self.name})"


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handles the OAuth callback request and stores the authorization code."""

    auth_code = None

    def do_GET(self):
        """Handle GET request, extract auth code from URL parameters."""
        query_components = parse_qs(urlparse(self.path).query)
        OAuthCallbackHandler.auth_code = query_components.get("code", [None])[0]

        # Send response to browser
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Authorization successful! You can close this window.")

        # Shutdown the server
        def shutdown():
            self.server.shutdown()

        Thread(target=shutdown).start()


def get_free_port() -> int:
    """Find a free port to use for the callback server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


async def get_auth_code(auth_url: str, port: int) -> str:
    """
    Start a local server to receive the OAuth callback and open the browser for authorization.
    Returns the authorization code.
    """
    # Start local server
    server = HTTPServer(("localhost", port), OAuthCallbackHandler)
    server_thread = Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Get the current status context if it exists
    status_context = getattr(console, "_live", None)

    # Pause status spinner if active
    if status_context:
        status_context.stop()

    # Ask for confirmation before opening browser
    console.print("\n[blue]I'll need to open your web browser to reconnect to Gmail.[/blue]")
    console.print("[dim]You'll be asked to sign in and grant permission to access your email.[/dim]")
    response = input("\nPress Enter to continue or type 'skip' to cancel this account: ")

    # Resume status spinner if it was active
    if status_context:
        status_context.start()

    if response.lower().strip() == "skip":
        server.shutdown()
        server.server_close()
        server_thread.join()
        raise ValueError("User canceled authentication process")

    # Open browser
    console.print("[green]Opening your browser...[/green]")
    webbrowser.open(auth_url)
    console.print("[yellow]Please complete the authentication in your browser.[/yellow]")

    # Wait for the authorization code
    waiting_dots = 0
    while OAuthCallbackHandler.auth_code is None:
        waiting_dots = (waiting_dots % 3) + 1
        console.print(f"[yellow]Waiting for authentication{('.' * waiting_dots):<3}[/yellow]", end="\r")
        await asyncio.sleep(1)

    console.print("\n[green]✓ Authentication successful![/green]")

    server.shutdown()
    server.server_close()
    server_thread.join()

    return OAuthCallbackHandler.auth_code


class AccountManager:
    def __init__(self, config_dir: str = "secrets"):
        self.config_dir = config_dir
        self.accounts_file = os.path.join(self.config_dir, "accounts.json")
        self.credentials_dir = os.path.join(self.config_dir, "credentials")
        self._ensure_dirs_exist()
        self._accounts: Dict[str, UserInfo] = {}
        self._load_accounts()

    def _ensure_dirs_exist(self) -> None:
        Path(self.config_dir).mkdir(parents=True, exist_ok=True)
        Path(self.credentials_dir).mkdir(parents=True, exist_ok=True)

    def _load_accounts(self) -> None:
        if os.path.exists(self.accounts_file):
            with open(self.accounts_file, "r") as f:
                data = json.load(f)
                self._accounts = {
                    shortname: UserInfo.from_dict(shortname, account_data) for shortname, account_data in data.items()
                }

    def _save_accounts(self) -> None:
        with open(self.accounts_file, "w") as f:
            data = {shortname: info.to_dict() for shortname, info in self._accounts.items()}
            json.dump(data, f, indent=2)

    def get_account_names(self) -> List[str]:
        return list(self._accounts.keys())

    def get_account(self, shortname: str) -> Optional[UserInfo]:
        return self._accounts.get(shortname)

    def add_account(self, shortname: str, email: str, name: str) -> UserInfo:
        user_info = UserInfo(shortname=shortname, name=name, email=email)
        self._accounts[shortname] = user_info
        self._save_accounts()
        return user_info

    async def select_account(self) -> Tuple[str, Aiogoogle]:
        """Interactive account selection. Returns a tuple (account_name, Gmail API service instance)."""
        accounts = self.get_account_names()

        if not accounts:
            print("No accounts configured. Let's set up your first account.")
            shortname = input("Enter a name for this account (e.g. 'work' or 'personal'): ").strip().lower()
            email = input("Enter the Gmail address: ").strip()
            name = input("Enter your name (for email classification): ").strip()
            self.add_account(shortname, email, name)
            return shortname, await self.get_service(shortname)

        if len(accounts) == 1:
            shortname = accounts[0]
            return shortname, await self.get_service(shortname)

        print("\nAvailable accounts:")
        for i, shortname in enumerate(accounts, 1):
            user_info = self._accounts[shortname]
            print(f"{i}. {user_info}")
        print(f"{len(accounts) + 1}. Add new account")

        while True:
            try:
                choice = int(input("\nSelect an account (enter number): "))
                if 1 <= choice <= len(accounts):
                    shortname = accounts[choice - 1]
                    return shortname, await self.get_service(shortname)
                elif choice == len(accounts) + 1:
                    shortname = input("Enter a name for the new account (e.g. 'work' or 'personal'): ").strip().lower()
                    email = input("Enter the Gmail address: ").strip()
                    name = input("Enter your name (for email classification): ").strip()
                    self.add_account(shortname, email, name)
                    return shortname, await self.get_service(shortname)
            except (ValueError, IndexError):
                print("Invalid choice. Please try again.")

    async def get_service(self, shortname: str) -> Aiogoogle:
        """Get Gmail API service for the specified account."""
        user_info = self.get_account(shortname)
        if user_info is None:
            logger.info("Account '%s' not found. Let's set it up.", shortname)
            email = input(f"Enter the Gmail address for '{shortname}': ").strip()
            name = input("Enter your name (for email classification): ").strip()
            user_info = self.add_account(shortname, email, name)

        token_path = self._get_token_path(user_info)
        credentials_path = os.path.join(self.credentials_dir, "application_credentials.json")

        logger.debug("Looking for credentials at: %s", credentials_path)
        logger.debug("Looking for token at: %s", token_path)

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"OAuth credentials not found. Please download the OAuth client credentials from Google Cloud Console "
                f"and save them to {credentials_path}"
            )

        async with aiofiles.open(credentials_path, "r") as f:
            content = await f.read()
            client_creds = json.loads(content)
            logger.debug("Loaded client credentials: %s", json.dumps(client_creds, indent=2))

        user_creds = None
        if os.path.exists(token_path):
            logger.debug("Found existing token file")
            async with aiofiles.open(token_path, "r") as f:
                content = await f.read()
                creds_data = json.loads(content)
                logger.debug("Loaded token data: %s", json.dumps(creds_data, indent=2))
                user_creds = UserCreds(
                    access_token=creds_data["access_token"],
                    refresh_token=creds_data["refresh_token"],
                    expires_at=creds_data["expiry"],
                    scopes=creds_data["scopes"],
                )
                logger.debug("Created UserCreds object with expiry: %s", user_creds.expires_at)
        else:
            logger.debug("No existing token file found")

        auth_creds = client_creds.get("installed", client_creds)
        auth_creds["scopes"] = SCOPES
        logger.debug("Using auth credentials: %s", json.dumps(auth_creds, indent=2))

        aiogoogle = Aiogoogle(client_creds=auth_creds, user_creds=user_creds)
        logger.debug(
            "Created initial Aiogoogle instance with user_creds: %s",
            "yes" if user_creds else "no",
        )

        if not user_creds:
            logger.debug("No user credentials found")
        elif not user_creds.valid:
            logger.debug("User credentials found but not valid")
            if user_creds.expired:
                logger.debug("Token is expired, expires_at: %s", user_creds.expires_at)
            else:
                logger.debug("Token is invalid for unknown reason")

        if not user_creds or not user_creds.valid:
            if user_creds and user_creds.expired and user_creds.refresh_token:
                try:
                    logger.debug("Attempting to refresh token")
                    user_creds = await aiogoogle.oauth2.refresh(user_creds)
                    logger.debug(
                        "Successfully refreshed token, new expiry: %s",
                        user_creds.expires_at,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to refresh token: {e}. Will need to reauthorize.",
                        exc_info=True,
                    )
                    user_creds = None

            if not user_creds:
                port = get_free_port()
                auth_creds["redirect_uri"] = f"http://localhost:{port}"
                logger.debug("Set up redirect URI: %s", auth_creds["redirect_uri"])

                auth_url = aiogoogle.oauth2.authorization_url(scopes=SCOPES)
                logger.debug("Generated auth URL: %s", auth_url)
                print(f"\nAuthorization required for Gmail access for account '{shortname}' ({user_info.email}).")
                input("Press Enter to open your browser for Gmail authorization...")
                code = await get_auth_code(auth_url, port)
                logger.debug("Received auth code")
                user_creds = await aiogoogle.oauth2.build_user_creds(code)
                logger.debug("Built new user credentials with expiry: %s", user_creds.expires_at)

            token_data = {
                "access_token": user_creds.access_token,
                "refresh_token": user_creds.refresh_token,
                "expiry": user_creds.expires_at,
                "scopes": user_creds.scopes,
            }
            async with aiofiles.open(token_path, "w") as f:
                await f.write(json.dumps(token_data, indent=2))
            logger.debug("Saved token data.")

        logger.debug(
            "Creating final Aiogoogle instance with user_creds: %s",
            "yes" if user_creds else "no",
        )
        return Aiogoogle(client_creds=auth_creds, user_creds=user_creds)

    def _get_token_path(self, user_info: UserInfo) -> str:
        """Return the path to the token file for a given user."""
        return os.path.join(self.credentials_dir, f"{user_info.email}_token.json")

    def _prompt_reauthorization(self, shortname: str, user_info: UserInfo) -> bool:
        """Prompt the user to decide if they want to reauthorize the account."""
        return (
            input(f"Would you like to reauthorize account '{shortname}' ({user_info.email}) now? (y/n): ")
            .lower()
            .strip()
            == "y"
        )

    async def _validate_account(self, shortname: str) -> None:
        """Validate a single account and handle reauthorization prompt on failure."""
        user_info = self.get_account(shortname)
        if not user_info:
            logger.error(f"Account '{shortname}' not found")
            return
        try:
            logger.info(f"Validating account '{shortname}' ({user_info.email})")
            await self.get_service(shortname)
            logger.info(f"Successfully validated account '{shortname}'")
        except Exception as e:
            logger.error(f"Failed to validate account '{shortname}': {e}")

            # Check if this is an authentication error
            if "invalid_grant" in str(e):
                console.print(
                    Panel.fit(
                        f"[yellow]Gmail authentication has expired for your [bold]{shortname}[/bold] account ({user_info.email}).[/yellow]\n"
                        f"[dim]This happens periodically and is normal.[/dim]",
                        title="[yellow]Authentication Needed[/yellow]",
                        border_style="yellow",
                    )
                )

                try:
                    console.print(f"[blue]Let's reconnect your [bold]{shortname}[/bold] account...[/blue]")
                    await self.reauthorize_account(shortname)
                    console.print(
                        Panel.fit(
                            f"[green]✓ Successfully reconnected to [bold]{shortname}[/bold]![/green]",
                            border_style="green",
                        )
                    )
                except ValueError as skip_err:
                    if "User canceled" in str(skip_err):
                        console.print(f"[yellow]Skipping account '{shortname}'[/yellow]")
                    else:
                        raise
                except Exception as re_auth_error:
                    logger.error(f"Failed to reauthorize account '{shortname}': {re_auth_error}")
                    console.print(
                        Panel.fit(
                            f"[red]I couldn't reconnect to your [bold]{shortname}[/bold] account.[/red]\n\n"
                            f"Error: {str(re_auth_error).strip()}\n\n"
                            f"[dim]Please try again later or check your network connection.[/dim]",
                            title="[red]⚠️ Reconnection Failed[/red]",
                            border_style="red",
                        )
                    )
            else:
                # For other errors
                console.print(
                    Panel.fit(
                        f"[red]There was a problem with your [bold]{shortname}[/bold] account ({user_info.email}).[/red]\n\n"
                        f"Error: {str(e).strip()}\n\n"
                        f"[dim]Please try again later or check your network connection.[/dim]",
                        title="[red]⚠️ Account Issue[/red]",
                        border_style="red",
                    )
                )

    async def validate_all_accounts(self) -> None:
        """Validate all configured accounts and refresh/reauthorize as needed."""
        logger.info("Validating all configured accounts...")
        console.print(Panel.fit("🔑 Checking your email accounts...", title="Zola", border_style="yellow"))

        accounts = self.get_account_names()
        with console.status("[bold green]Connecting to your accounts...") as status:
            for shortname in accounts:
                await self._validate_account(shortname)

        console.print(
            Panel.fit(
                f"[green]✓[/green] Connected to {len(accounts)} email account{'s' if len(accounts) != 1 else ''}",
                title="Zola",
                border_style="green",
            )
        )
        logger.info("Finished validating all accounts")

    async def reauthorize_account(self, shortname: str) -> None:
        """Explicitly reauthorize an account by clearing existing credentials and starting fresh."""
        user_info = self.get_account(shortname)
        if not user_info:
            raise ValueError(f"Account '{shortname}' not found")

        token_path = self._get_token_path(user_info)
        if os.path.exists(token_path):
            logger.info(f"Removing existing token file for '{shortname}'")
            os.remove(token_path)

        logger.info(f"Starting reauthorization flow for '{shortname}' ({user_info.email})")
        await self.get_service(shortname)  # This will trigger the full auth flow
        logger.info(f"Successfully reauthorized account '{shortname}'")
