# Zola: Local Email Assistant

Zola is an experiment that uses a local large language model (LLM) to read and classify your Gmail emails. It runs entirely on your computer. The only time it connects to the internet is to fetch your emails from Gmail—your email data never leaves your machine.

It might crash, it might not work, it might be slow. It will definately heat up your computer. It’s early.

This project depends on [mlx](https://github.com/ml-explore/mlx) and only works on Apple Silicon Macs.

## What It Does

- Reads emails from your Inbox and Sent folder: Zola reads `--max-messages` from your Sent folder to see who is important to you, and from your Inbox to classify.
- Organizes, not changes: Zola only adds labels in a dedicated Zola subfolder. It doesn’t mark emails as read or delete them for now.
- Local and private: Everything happens on your machine. No email data is sent to a remote service.
- Smart classification: Zola uses a local LLM to decide what labels to assign based on each email’s content and urgency. For example, it might label an email with `["priority", "finance"]`.

## How It Works

  1. Fetch emails: It uses the Gmail API to get your messages.
  2. Convert content: Email HTML and attachments are converted to Markdown using the markitdown library.
  3. Build context: It gathers email details like metadata, and historical reply times.
  4. Classify: The LLM reads this context and returns one or more classification labels.

## Feedback and Customization

This is early! The easiest way to help is by suggesting a labeling system and prompt improvements. These are in the [`settings.py`](zola/settings.py) file. This file sets up the filing system and the instructions (the system prompt) that guide the LLM’s classifications.

## Future

- [ ] Make it better
- [ ] Add a UI

## Installation

Zola is managed with [uv from Astral](https://github.com/astral-sh/uv) (`brew install uv`). To install the dependencies, clone the repository and run uv:

```bash
git clone https://github.com/hartshorne/zola.git
cd zola
uv sync
```

## Setup

### Gmail API Credentials

Zola requires OAuth 2.0 client credentials to access your Gmail account. You can create your own test credentials by following these steps:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Enable the Gmail API for your project by visiting the [Gmail API page](https://console.cloud.google.com/marketplace/product/google/gmail.googleapis.com).
4. Create OAuth 2.0 client credentials by going to the [Credentials page](https://console.cloud.google.com/apis/credentials):
   - Select **Create credentials**.
   - Choose **OAuth client ID**.
   - Select **Desktop app** as the application type.
   - Name the client ID (for example, "Zola").
5. Download the OAuth client credentials JSON file, and move it to `secrets/credentials/application_credentials.json`.

### Account Configuration

Zola supports multiple Gmail accounts. When you first run Zola, it will:

1. Create a configuration directory at `secrets/` in the project root
2. Ask you to name your account (e.g., "work" or "personal")
3. Ask for the Gmail address associated with the account
4. Open a browser window for you to authorize the application with your Gmail account
5. Save the authorization token for future use at `secrets/credentials/{email}_credentials.json`

You can add more accounts later by using the `--account` flag with a new account name.

## Labels

Zola uses a hierarchical labeling system to organize your inbox. All labels are prefixed with `zola/`:

### Standard Labels

- **priority**: For emails needing immediate attention, such as urgent messages, near-due bills, or emails from key contacts.
- **respond**: For emails that require a reply or follow-up. Can also be marked as `priority` if urgent.
- **review**: For emails to be read or reviewed later; not urgent and not requiring a response.
- **finance**: For financial matters like invoices, bills, expenses, and taxes. Can be marked as `priority` if urgent.
- **travel**: For travel-related emails, including bookings, itineraries, and confirmations. Use `priority` if time-sensitive.
- **school**: For school communications, such as updates, announcements, and deadlines. Mark as `priority` if urgent.
- **newsletters**: For regular updates like newsletters and informational digests.
- **ignore**: For low-priority bulk emails, including sales outreach, surveys, promotional content, or automated notifications. **When this label is applied, no other labels should be used.**

### System Labels

System labels are prefixed with `zola/🤖/` and indicate processing status:

- **error**: Applied when LLM processing fails with an exception
- **confused**: Applied when the LLM fails to classify the email

### Response Time Categories

Zola tracks your typical response times to help prioritize incoming emails:

- **immediate**: within 30 minutes
- **quick**: within 2 hours
- **same_day**: within 8 hours
- **next_day**: within 24 hours
- **delayed**: longer than 24 hours

These categories are based on your historical reply times per sender and domain, assisting the LLM in assessing the urgency of each email.

## Usage

When you run Zola for the first time, it will download the language model you choose from [Hugging Face](https://huggingface.co/) if it isn’t already cached on your computer. You might need to set up a [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) in your environment to access certain models (for example, private or gated ones). Models are big, and the default cache location is `~/.cache/huggingface/hub/`.

Zola provides a command-line interface with the following options:

- `--max-messages, -m`: Set the maximum number of messages to process (default: 10).
- `--model, -M`: Specify the LLM model identifier/path to use. This argument defaults to `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` and accepts one of the following values:
  - `mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit`
  - `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`
  - `mlx-community/Mistral-Small-24B-Instruct-2501-8bit`
  - `mlx-community/Llama-3.3-70B-Instruct-8bit`
- `--account, -a`: Specify which account to use (if not provided, will process all configured accounts)
- `--log-level, -l`: Set the logging level (default: INFO)
- `--skip-llm-test, -s`: Skip the LLM test at startup

Zola will:

1. Process Sent messages to build SLA (reply time) data and save average SLA per domain to `secrets/sla/{account_name}.json`.
2. Process Inbox messages and apply labels based on the LLM’s output and SLA adjustments.

## Running Zola

Run the module using:

```bash
python -m zola.main --max-messages 500  # Process all accounts
```

To use a specific account:

```bash
python -m zola.main --account work
```

## Testing

Unit tests are provided in the `tests/` directory. To run the tests, execute:

```bash
python -m unittest discover -s tests
```
