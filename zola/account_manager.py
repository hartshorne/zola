import asyncio
import json
import logging
import os
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import parse_qs, urlparse

from aiogoogle.client import Aiogoogle
from aiogoogle.auth.creds import UserCreds

from zola.settings import SCOPES

logger = logging.getLogger(__name__)


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

    # Open browser
    webbrowser.open(auth_url)

    # Wait for the authorization code
    while OAuthCallbackHandler.auth_code is None:
        await asyncio.sleep(1)

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
        self.accounts = self._load_accounts()

    def _ensure_dirs_exist(self) -> None:
        Path(self.config_dir).mkdir(parents=True, exist_ok=True)
        Path(self.credentials_dir).mkdir(parents=True, exist_ok=True)

    def _load_accounts(self) -> Dict[str, str]:
        if os.path.exists(self.accounts_file):
            with open(self.accounts_file, "r") as f:
                return json.load(f)
        return {}

    def _save_accounts(self) -> None:
        with open(self.accounts_file, "w") as f:
            json.dump(self.accounts, f, indent=2)

    def get_account_names(self) -> list[str]:
        return list(self.accounts.keys())

    def add_account(self, name: str, email: str, user_name: str) -> None:
        self.accounts[name] = {"email": email, "name": user_name}
        self._save_accounts()

    async def select_account(self) -> Tuple[str, Aiogoogle]:
        """
        Interactive account selection.
        Returns a tuple (account_name, Gmail API service instance).
        """
        accounts = self.get_account_names()

        if not accounts:
            print("No accounts configured. Let's set up your first account.")
            name = input("Enter a name for this account (e.g. 'work' or 'personal'): ").strip()
            email = input("Enter the Gmail address: ").strip()
            user_name = input("Enter your name (for email classification): ").strip()
            self.add_account(name, email, user_name)
            return name, await self.get_service(name)

        if len(accounts) == 1:
            name = accounts[0]
            return name, await self.get_service(name)

        print("\nAvailable accounts:")
        for i, name in enumerate(accounts, 1):
            print(f"{i}. {name} ({self.accounts[name]['email']} - {self.accounts[name]['name']})")
        print(f"{len(accounts) + 1}. Add new account")

        while True:
            try:
                choice = int(input("\nSelect an account (enter number): "))
                if 1 <= choice <= len(accounts):
                    name = accounts[choice - 1]
                    return name, await self.get_service(name)
                elif choice == len(accounts) + 1:
                    name = input("Enter a name for the new account (e.g. 'work' or 'personal'): ").strip()
                    email = input("Enter the Gmail address: ").strip()
                    user_name = input("Enter your name (for email classification): ").strip()
                    self.add_account(name, email, user_name)
                    return name, await self.get_service(name)
            except (ValueError, IndexError):
                print("Invalid choice. Please try again.")

    async def get_service(self, account_name: str) -> Aiogoogle:
        """
        Get Gmail API service for the specified account.
        If the account doesn't exist, guides the user through creating it.
        Loads stored token if it exists, otherwise initiates OAuth flow.
        """
        if account_name not in self.accounts:
            logger.info("Account '%s' not found. Let's set it up.", account_name)
            email = input(f"Enter the Gmail address for '{account_name}': ").strip()
            user_name = input("Enter your name (for email classification): ").strip()
            self.add_account(account_name, email, user_name)

        account_info = self.accounts[account_name]
        email = account_info["email"]
        token_path = os.path.join(self.credentials_dir, f"{email}_token.json")
        credentials_path = os.path.join(self.credentials_dir, "application_credentials.json")

        logger.debug("Looking for credentials at: %s", credentials_path)
        logger.debug("Looking for token at: %s", token_path)

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"OAuth credentials not found. Please download the OAuth client credentials from Google Cloud Console "
                f"and save them to {credentials_path}"
            )

        with open(credentials_path) as f:
            client_creds = json.load(f)
            logger.debug("Loaded client credentials: %s", json.dumps(client_creds, indent=2))

        user_creds = None
        if os.path.exists(token_path):
            logger.debug("Found existing token file")
            with open(token_path) as f:
                creds_data = json.load(f)
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
                print(f"\nAuthorization required for Gmail access for account '{account_name}' ({email}).")
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
            logger.debug("Saving token data: %s", json.dumps(token_data, indent=2))
            with open(token_path, "w") as f:
                json.dump(token_data, f, indent=2)

        logger.debug(
            "Creating final Aiogoogle instance with user_creds: %s",
            "yes" if user_creds else "no",
        )
        return Aiogoogle(client_creds=auth_creds, user_creds=user_creds)
