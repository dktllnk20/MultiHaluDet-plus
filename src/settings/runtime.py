"""Provides shared filesystem and logging helpers for the pipeline.
run_pipeline.py uses these helpers to manage output folders and capture run logs.
Feature and result files remain in their existing project-level output directories.
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys
import traceback
import warnings

from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.theme import Theme


ROOT_DIR = Path(__file__).resolve().parents[2]


def ensure_directory(path):
    """Create a directory and return its Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent(path):
    """Create the parent directory for a file and return its Path."""
    file_path = Path(path)
    ensure_directory(file_path.parent)
    return file_path


class _PipelineHighlighter(RegexHighlighter):
    highlights = [
        r"(?i)(?P<error>\b(?:error|failed|failure|exception|traceback)\b)",
        r"(?i)(?P<warning>\b(?:warning|warn)\b)",
        r"(?i)(?P<success>\b(?:success|completed|saved|done)\b)",
        r"(?i)(?P<stage>\bstage\s+\d+|---[^-\n]+---)",
        r"(?i)(?P<metric>\b(?:auc|f1|accuracy|seed|dataset|model):)",
        r"(?P<number>(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\w.]))",
    ]


class _RichTerminal:
    def __init__(self, stream):
        self.stream = stream
        self.console = Console(
            file=stream,
            force_terminal=stream.isatty(),
            highlight=True,
            highlighter=_PipelineHighlighter(),
            theme=Theme({
                "error": "bold red",
                "warning": "bold yellow",
                "success": "bold green",
                "stage": "bold cyan",
                "metric": "bold magenta",
                "number": "bold bright_blue",
            }),
            soft_wrap=True,
        )

    def write(self, text):
        if text:
            self.console.print(text, end="", markup=False, highlight=True)
        return len(text)

    def flush(self):
        self.console.file.flush()

    def isatty(self):
        return self.stream.isatty()

    @property
    def encoding(self):
        return self.stream.encoding


class _Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    @property
    def encoding(self):
        return self.terminal.encoding


@contextmanager
def capture_run_log(path):
    """Append a timestamped run section and display highlighted output without terminal warnings."""
    log_path = ensure_parent(path)
    terminal_out, terminal_err = sys.stdout, sys.stderr
    has_previous_run = log_path.exists() and log_path.stat().st_size > 0
    with log_path.open("a", encoding="utf-8") as log:
        if has_previous_run:
            log.write("\n")
        separator = "-" * 28
        log.write(f"{separator}\n")
        log.write(f"Run started: {datetime.now().strftime('%d-%m-%Y (%H:%M:%S)')}\n")
        log.write(f"{separator}\n")
        log.flush()
        rich_out = _RichTerminal(terminal_out)
        with redirect_stdout(_Tee(rich_out, log)), redirect_stderr(_Tee(terminal_err, log)):
            original_showwarning = warnings.showwarning

            def log_warning(message, category, filename, lineno, file=None, line=None):
                log.write(warnings.formatwarning(message, category, filename, lineno, line=line))
                log.flush()

            warnings.showwarning = log_warning
            try:
                yield log_path
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                warnings.showwarning = original_showwarning
