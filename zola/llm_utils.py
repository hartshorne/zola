import ast
import asyncio
import logging
import threading
from typing import Any, List, Tuple

from mlx_lm.utils import generate as mlx_generate
from mlx_lm.utils import load as mlx_load
from rich.panel import Panel

from zola.console import console
from zola.settings import LLM_PARENT_LABEL, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, PARENT_LABEL, STANDARD_LABELS

logger = logging.getLogger(__name__)

# Global lock to protect the LLM from being called concurrently from multiple threads.
_LLM_LOCK = threading.Lock()


class SafeTokenizer:
    """Wrapper around the tokenizer to ensure reasonable context lengths."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    @property
    def model_max_length(self) -> int:
        """Return a reasonable context length, capped at MAX_INPUT_TOKENS."""
        raw_length = getattr(self._tokenizer, "model_max_length", MAX_INPUT_TOKENS)
        return min(raw_length, MAX_INPUT_TOKENS)

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying tokenizer."""
        return getattr(self._tokenizer, name)


def log_thread_pool_usage():
    active_count = threading.active_count()
    thread_names = [t.name for t in threading.enumerate()]
    logger.info(
        "Thread pool usage: active threads=%d, thread names=%s",
        active_count,
        thread_names,
    )


def load_llm(model_identifier: str) -> Tuple[Any, Any]:
    """
    Loads the local LLM using mlx-llm.
    Returns the model and a wrapped tokenizer with sane context lengths.
    """
    model, raw_tokenizer = mlx_load(model_identifier)
    tokenizer = SafeTokenizer(raw_tokenizer)

    # Get the raw context length for comparison
    raw_length = getattr(raw_tokenizer, "model_max_length", MAX_INPUT_TOKENS)
    actual_length = tokenizer.model_max_length

    # Print model statistics
    console.print(
        Panel.fit(
            f"[bold blue]Model Information[/bold blue]\n\n"
            f"Model: [cyan]{model_identifier}[/cyan]\n"
            f"Context Window: [cyan]{actual_length:,}[/cyan] tokens"
            + (f" (capped from {raw_length:,})" if raw_length > actual_length else "")
            + "\n"
            f"Vocab Size: [cyan]{len(tokenizer.vocab):,}[/cyan] tokens",
            title="[bold blue]Model Statistics[/bold blue]",
            border_style="blue",
        )
    )

    return model, tokenizer


def safe_mlx_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    verbose: bool = False,
) -> str:
    """
    Calls mlx_generate while acquiring the lock so that only one call occurs at a time.
    """
    thread_name = threading.current_thread().name
    logger.info("Thread %s: Attempting to acquire LLM lock", thread_name)

    # Log thread pool usage before trying to acquire the lock
    log_thread_pool_usage()

    _LLM_LOCK.acquire()
    try:
        logger.info("Thread %s: Acquired LLM lock", thread_name)
        result = mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            verbose=verbose,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        logger.info("Thread %s: LLM generate call completed", thread_name)
        return result
    finally:
        _LLM_LOCK.release()
        logger.info("Thread %s: Released LLM lock", thread_name)


async def classify_email(context: str, model: Any, tokenizer: Any) -> List[str]:
    """
    Uses the local LLM (via mlx-llm) to classify the email based on the provided context.
    Returns a list of labels. Standard labels are returned without prefix (e.g. "priority"),
    while error labels are returned with full path (using parent label prefix).
    """
    try:
        # Log the token count before sending to LLM
        context_tokens = tokenizer.encode(context)
        token_count = len(context_tokens)
        logger.info(
            "Sending %d tokens to LLM (%.1f%% of model's limit)",
            token_count,
            (token_count / tokenizer.model_max_length) * 100,
        )

        # Log thread pool usage before dispatching the LLM call in a new thread.
        log_thread_pool_usage()

        # Use asyncio.to_thread to call the thread-safe generate function.
        result: str = await asyncio.to_thread(safe_mlx_generate, model, tokenizer, context, False)
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return [f"{LLM_PARENT_LABEL}/error"]

    # Parse the result to find a valid list
    lines = result.strip().split("\n")
    lines.reverse()
    list_content = ""
    list_started = False
    list_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "[" in line and "]" in line:
            list_lines.append(line)
            list_started = True
        elif list_started and not line.endswith("]"):
            list_lines.append(line)

    list_lines.reverse()
    for line in list_lines:
        try:
            start = line.find("[")
            end = line.rfind("]") + 1
            if start != -1 and end != -1:
                list_content = line[start:end]
                output = ast.literal_eval(list_content)
                if isinstance(output, list) and all(isinstance(item, str) for item in output):
                    # Strip any existing prefixes for standard labels
                    valid_labels = []
                    for label in output:
                        label = label.lower()
                        # Remove any existing prefix for standard labels
                        if label.startswith(f"{PARENT_LABEL}/"):
                            label = label[len(f"{PARENT_LABEL}/") :]
                        if label in STANDARD_LABELS:
                            valid_labels.append(label)

                    if valid_labels:
                        logger.info(
                            "Successfully parsed list from LLM output with valid labels: %s",
                            valid_labels,
                        )
                        return valid_labels
                    else:
                        logger.warning(
                            "LLM returned labels but none were in STANDARD_LABELS: %s",
                            output,
                        )
                        return [f"{LLM_PARENT_LABEL}/confused"]
        except (SyntaxError, ValueError) as e:
            logger.warning("Failed to parse potential list %s: %s", line, e)
            continue

    logger.warning("No valid list found in LLM output: %s", result)
    return [f"{LLM_PARENT_LABEL}/confused"]
