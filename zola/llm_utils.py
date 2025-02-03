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
_LLM_LOCK = threading.Lock()


class SafeTokenizer:
    """Wrapper around the tokenizer to ensure reasonable context lengths."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    @property
    def model_max_length(self) -> int:
        raw_length = getattr(self._tokenizer, "model_max_length", MAX_INPUT_TOKENS)
        return min(raw_length, MAX_INPUT_TOKENS)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


def log_thread_pool_usage() -> None:
    active_count = threading.active_count()
    thread_names = [t.name for t in threading.enumerate()]
    logger.info("Thread pool usage: active threads=%d, thread names=%s", active_count, thread_names)


def load_llm(model_identifier: str) -> Tuple[Any, Any]:
    model, raw_tokenizer = mlx_load(model_identifier)
    tokenizer = SafeTokenizer(raw_tokenizer)
    raw_length = getattr(raw_tokenizer, "model_max_length", MAX_INPUT_TOKENS)
    actual_length = tokenizer.model_max_length

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


def safe_mlx_generate(model: Any, tokenizer: Any, prompt: str, verbose: bool = False) -> str:
    thread_name = threading.current_thread().name
    logger.info("Thread %s: Attempting to acquire LLM lock", thread_name)
    log_thread_pool_usage()
    with _LLM_LOCK:
        logger.info("Thread %s: Acquired LLM lock", thread_name)
        result = mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            verbose=verbose,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        logger.info("Thread %s: LLM generate call completed", thread_name)
    logger.info("Thread %s: Released LLM lock", thread_name)
    return result


def parse_llm_output(result: str) -> List[str]:
    """Extract a list from the LLM output if present."""
    for line in result.splitlines():
        line = line.strip()
        if "[" in line and "]" in line:
            try:
                start, end = line.find("["), line.rfind("]") + 1
                candidate = line[start:end]
                output = ast.literal_eval(candidate)
                if isinstance(output, list) and all(isinstance(item, str) for item in output):
                    return output
            except (SyntaxError, ValueError):
                logger.warning("Failed to parse candidate list: %s", line)
    return []


def normalize_labels(raw_labels: List[str]) -> List[str]:
    valid_labels = []
    for label in raw_labels:
        norm = label.lower().removeprefix(f"{PARENT_LABEL}/")
        if norm in STANDARD_LABELS:
            valid_labels.append(norm)
    return valid_labels


async def classify_email(context: str, model: Any, tokenizer: Any) -> List[str]:
    try:
        context_tokens = tokenizer.encode(context)
        token_count = len(context_tokens)
        logger.info(
            "Sending %d tokens to LLM (%.1f%% of model's limit)",
            token_count,
            (token_count / tokenizer.model_max_length) * 100,
        )
        log_thread_pool_usage()
        result: str = await asyncio.to_thread(safe_mlx_generate, model, tokenizer, context, False)
    except Exception:
        logger.exception("LLM generation failed")
        return [f"{LLM_PARENT_LABEL}/error"]

    raw_labels = parse_llm_output(result)
    valid_labels = normalize_labels(raw_labels)
    if valid_labels:
        logger.info("Successfully parsed valid labels: %s", valid_labels)
        return valid_labels
    else:
        logger.warning("No valid labels found in LLM output: %s", result)
        return [f"{LLM_PARENT_LABEL}/confused"]
