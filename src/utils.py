"""
Utility functions for console formatting and display.
Provides consistent, professional output throughout the pipeline.
"""

import sys
import io

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


def print_header(text):
    """Print a major section header with double-line border."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_subheader(text):
    """Print a subsection header with single-line border."""
    print(f"\n{'-' * 55}")
    print(f"  {text}")
    print(f"{'-' * 55}")


def print_step(step_num, text):
    """Print a numbered pipeline step."""
    print(f"\n  [Step {step_num}] {text}")


def print_info(text):
    """Print an informational message."""
    print(f"  [INFO] {text}")


def print_success(text):
    """Print a success message."""
    print(f"  [OK]   {text}")


def print_metric(name, value):
    """Print a formatted metric key-value pair."""
    print(f"  |-- {name}: {value}")


def print_separator():
    """Print a simple separator line."""
    print("-" * 55)


def print_banner():
    """Print the project banner on startup."""
    banner = r"""
    +==============================================================+
    |     ADVANCED SENTIMENT ANALYSIS SYSTEM                       |
    |     Machine Learning & Deep Learning Pipeline                |
    |     -----------------------------------------                |
    |     IMDB Movie Reviews | 5 Models | Full Evaluation          |
    +==============================================================+
    """
    print(banner)
