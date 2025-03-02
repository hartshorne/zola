# Zola: Local Email Assistant

Zola is a local assistant that uses a large language model (LLM) to read and classify your Gmail messages. It runs entirely on your Mac, connecting to the internet only to fetch emails from Gmail—your data never leaves your machine.

It might crash, it might not work, it might be slow. It will definately heat up your computer. It’s early.

Zola requires [mlx](https://github.com/ml-explore/mlx) and works only on Apple Silicon Macs.

## What Zola Does

- **Reads Your Emails:** Zola scans your Inbox and Sent folders. It uses your Sent messages to identify important contacts and your Inbox messages for classification.
- **Organizes without Altering:** Zola only labels emails within a special Zola subfolder. It doesn't mark emails as read or delete anything.
- **Local and Private:** All processing happens on your Mac. No email data is sent to remote servers.
- **Smart Classification:** The local LLM analyzes each email’s content and urgency, assigning labels like `"priority"` or `"finance"`.

## How Zola Works

1. **Fetch Emails:** Retrieves messages using Gmail’s API.
2. **Convert Content:** Transforms email HTML and attachments into Markdown using the `markitdown` library.
3. **Contextualize:** Gathers metadata and your historical reply times to understand each email’s context.
4. **Classify:** The LLM assigns relevant labels based on this context.

## Feedback and Customization

Zola is still developing. The best way to help is by suggesting improvements to its labeling system and prompts. Customize these in the [`settings.py`](zola/settings.py) file, which controls label structure and LLM behavior.

## Future

- [ ] Make it better
- [ ] Add a UI

## Installation

Zola uses [uv from Astral](https://github.com/astral-sh/uv). To install, first set up `uv`:

```bash
brew install uv
```

Then clone the repository and install dependencies:

```bash
git clone https://github.com/hartshorne/zola.git
cd zola
uv sync
```

## Setup

### Gmail API Credentials

Zola needs OAuth 2.0 credentials to access Gmail:

1. Visit [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project.
3. Enable the Gmail API via [Gmail API page](https://console.cloud.google.com/marketplace/product/google/gmail.googleapis.com).
4. Create OAuth credentials:
   - Go to [Credentials page](https://console.cloud.google.com/apis/credentials).
   - Click **Create credentials**, choose **OAuth client ID**, select **Desktop app**, and name it (e.g., “Zola”).
5. Download and save the JSON credentials to `secrets/credentials/application_credentials.json`.

## Usage

Running Zola for the first time will download the required LLM model from [Hugging Face](https://huggingface.co/). You might need a [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) for some models. The default model cache is at `~/.cache/huggingface/hub/`.

Available command-line options:

- `--max-messages, -m`: Number of messages to process (default: 10).
- `--model, -M`: Choose the LLM model. Defaults to `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` and accepts:
  - `mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit`
  - `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`
  - `mlx-community/Mistral-Small-24B-Instruct-2501-8bit`
  - `mlx-community/Llama-3.3-70B-Instruct-8bit`
- `--account, -a`: Select a specific Gmail account (defaults to all configured accounts).
- `--log-level, -l`: Logging verbosity (default: INFO).
- `--skip-llm-test, -s`: Skip initial LLM tests.

Zola's workflow:

1. Reads Sent messages to calculate average reply times (SLA), saving results to `secrets/sla/{account_name}.json`.
2. Reads Inbox messages, applying labels according to the LLM’s suggestions and SLA data.

## Running Zola

Process all accounts:

```bash
python -m zola.main --max-messages 500
```

Process a specific account:

```bash
python -m zola.main --account work
```

## Testing

Run unit tests with:

```bash
python -m unittest discover -s tests
```
