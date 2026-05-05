#!/usr/bin/env python3
"""Branch summary and timeline automation for repo-local git hooks."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BRANCH_DIR = REPO_ROOT / "docs" / "branch"
TIMELINE_PATH = BRANCH_DIR / "BRANCH_TIMELINE.md"
TEMPLATE_PATH = BRANCH_DIR / "BRANCH_SUMMARY_TEMPLATE.md"
MAIN_BRANCHES = {"main", "master"}

REQUIRED_METADATA = [
    "Branch",
    "Status",
    "Opened on",
    "Closed on",
    "Merged into",
    "Merge strategy",
    "Last updated",
]

REQUIRED_SECTIONS = [
    "Accomplishments",
    "Planned work",
    "Executed work",
    "Back-and-forth / iteration notes",
    "Problems + resolutions",
    "Validation",
    "Final changelog-style outcome",
]


def run_git(args: List[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def current_branch() -> str:
    return run_git(["branch", "--show-current"])


def sanitize_branch_name(branch: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", branch)


def summary_path_for_branch(branch: str) -> pathlib.Path:
    return BRANCH_DIR / f"BRANCH_SUMMARY_{sanitize_branch_name(branch)}.md"


def today_str() -> str:
    return dt.date.today().isoformat()


def parse_date(value: str) -> Optional[dt.date]:
    value = value.strip()
    if not value or value == "-":
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_metadata(content: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for line in content.splitlines():
        m = re.match(r"^\s*-\s+\*\*([^*]+)\*\*:\s*(.*)\s*$", line)
        if not m:
            continue
        metadata[m.group(1).strip()] = m.group(2).strip()
    return metadata


def has_heading(content: str, heading: str) -> bool:
    pattern = rf"^\s*##\s+{re.escape(heading)}\s*$"
    return re.search(pattern, content, flags=re.MULTILINE) is not None


def accomplishments_bullets(content: str) -> List[str]:
    heading_match = re.search(r"^\s*##\s+Accomplishments\s*$", content, flags=re.MULTILINE)
    if not heading_match:
        return []
    rest = content[heading_match.end() :]
    next_heading = re.search(r"^\s*##\s+", rest, flags=re.MULTILINE)
    section = rest if not next_heading else rest[: next_heading.start()]
    bullets = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            bullets.append(line[2:].strip())
    return [b for b in bullets if b]


def render_summary_template(branch: str) -> str:
    today = today_str()
    return f"""# Branch Summary: {branch}

## Metadata
- **Branch**: {branch}
- **Status**: open
- **Opened on**: {today}
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
- **Last updated**: {today}

## Accomplishments
- Add 2-5 key accomplishments before committing.
- Keep each bullet concise and outcome-focused.

## Planned work
- 

## Executed work
- 

## Back-and-forth / iteration notes
- 

## Problems + resolutions
- 

## Validation
- 

## Final changelog-style outcome
- 
"""


def ensure_template_shape() -> None:
    content = render_summary_template("<branch-name>")
    write_text(TEMPLATE_PATH, content)


def ensure_branch_summary(branch: str) -> pathlib.Path:
    BRANCH_DIR.mkdir(parents=True, exist_ok=True)
    ensure_template_shape()
    path = summary_path_for_branch(branch)
    if not path.exists():
        write_text(path, render_summary_template(branch))
        return path

    content = read_text(path)
    metadata = parse_metadata(content)
    updated = content

    if metadata.get("Branch") != branch:
        updated = re.sub(
            r"^(\s*-\s+\*\*Branch\*\*:\s*).*$",
            rf"\1{branch}",
            updated,
            flags=re.MULTILINE,
        )

    defaults = {
        "Status": "open",
        "Opened on": today_str(),
        "Closed on": "-",
        "Merged into": "-",
        "Merge strategy": "-",
        "Last updated": today_str(),
    }
    for key, value in defaults.items():
        if key not in metadata:
            updated = updated.replace(
                "## Accomplishments",
                f"- **{key}**: {value}\n\n## Accomplishments",
                1,
            )

    for section in REQUIRED_SECTIONS:
        if not has_heading(updated, section):
            updated = updated.rstrip() + f"\n\n## {section}\n- \n"

    if updated != content:
        write_text(path, updated)
    return path


def update_metadata_value(content: str, key: str, value: str) -> str:
    pattern = rf"^(\s*-\s+\*\*{re.escape(key)}\*\*:\s*).*$"
    # Use \g<1> so the next character is not parsed as part of a numeric backreference
    # (e.g. "\12026-05-05" would treat \120 as octal and corrupt the line).
    replacement = r"\g<1>" + value
    if re.search(pattern, content, flags=re.MULTILINE):
        return re.sub(pattern, replacement, content, flags=re.MULTILINE)
    return content


def validate_summary(path: pathlib.Path, set_last_updated: bool) -> Tuple[bool, List[str], str]:
    errors: List[str] = []
    content = read_text(path)
    metadata = parse_metadata(content)

    for key in REQUIRED_METADATA:
        if key not in metadata:
            errors.append(f"Missing metadata field: {key}")

    for section in REQUIRED_SECTIONS:
        if not has_heading(content, section):
            errors.append(f"Missing required section: {section}")

    status = metadata.get("Status", "").strip()
    if status not in {"open", "closed"}:
        errors.append("Status must be 'open' or 'closed'.")

    opened_on = metadata.get("Opened on", "")
    if parse_date(opened_on) is None:
        errors.append("Opened on must be YYYY-MM-DD.")

    last_updated = metadata.get("Last updated", "")
    if set_last_updated:
        content = update_metadata_value(content, "Last updated", today_str())
        metadata["Last updated"] = today_str()
        last_updated = today_str()
    elif parse_date(last_updated) is None:
        errors.append("Last updated must be YYYY-MM-DD.")

    closed_on = metadata.get("Closed on", "-")
    if closed_on != "-" and parse_date(closed_on) is None:
        errors.append("Closed on must be '-' or YYYY-MM-DD.")

    merged_into = metadata.get("Merged into", "-").strip()
    merge_strategy = metadata.get("Merge strategy", "-").strip()
    if not merged_into:
        errors.append("Merged into must be '-' or a branch name.")
    if not merge_strategy:
        errors.append("Merge strategy must be '-' or a strategy value.")

    if status == "closed" and closed_on == "-":
        errors.append("Closed branches must include Closed on date.")

    bullets = accomplishments_bullets(content)
    if not (2 <= len(bullets) <= 5):
        errors.append("Accomplishments must contain 2-5 bullets.")

    return (len(errors) == 0, errors, content)


def discover_summaries() -> List[pathlib.Path]:
    if not BRANCH_DIR.exists():
        return []
    summaries = []
    for path in BRANCH_DIR.glob("BRANCH_SUMMARY_*.md"):
        if path.name == "BRANCH_SUMMARY_TEMPLATE.md":
            continue
        summaries.append(path)
    return sorted(summaries)


def branch_name_from_summary(path: pathlib.Path, content: str) -> str:
    metadata = parse_metadata(content)
    if metadata.get("Branch"):
        return metadata["Branch"]
    # Fallback if metadata missing.
    stem = path.stem.replace("BRANCH_SUMMARY_", "")
    return stem


def timeline_entry(path: pathlib.Path) -> Optional[Dict[str, str]]:
    content = read_text(path)
    if not content.strip():
        return None
    metadata = parse_metadata(content)
    branch = branch_name_from_summary(path, content)
    accomplishments = accomplishments_bullets(content)[:3]
    opened = metadata.get("Opened on", "-")
    last_updated = metadata.get("Last updated", "-")
    closed_on = metadata.get("Closed on", "-")
    merged_into = metadata.get("Merged into", "-")
    merge_strategy = metadata.get("Merge strategy", "-")
    closed_col = closed_on if closed_on != "-" else "open"
    rel_summary = path.relative_to(REPO_ROOT).as_posix()
    return {
        "branch": branch,
        "opened": opened,
        "last_updated": last_updated,
        "closed": closed_col,
        "merged_into": merged_into,
        "merge_strategy": merge_strategy,
        "summary_link": f"[{path.name}]({rel_summary})",
        "accomplishments": accomplishments,
    }


def timeline_sort_key(entry: Dict[str, str]) -> Tuple[dt.date, str]:
    parsed = parse_date(entry.get("last_updated", ""))
    if parsed is None:
        parsed = dt.date.min
    return (parsed, entry.get("branch", ""))


def regenerate_timeline() -> None:
    BRANCH_DIR.mkdir(parents=True, exist_ok=True)
    ensure_template_shape()
    entries = []
    for summary in discover_summaries():
        entry = timeline_entry(summary)
        if entry:
            entries.append(entry)
    entries.sort(key=timeline_sort_key, reverse=True)

    lines: List[str] = [
        "# Branch Timeline",
        "",
        "Generated by `scripts/branch_docs.py`.",
        "",
    ]
    if not entries:
        lines.extend(["No branch summaries found.", ""])
    else:
        for entry in entries:
            lines.append(f"### {entry['branch']}")
            lines.append("")
            lines.append("**Accomplishments**")
            if entry["accomplishments"]:
                for bullet in entry["accomplishments"]:
                    lines.append(f"- {bullet}")
            else:
                lines.append("- (no accomplishments listed)")
            lines.append("")
            lines.append(
                "| Opened | Last Updated | Closed | Merged Into | Merged Strategy | Summary |"
            )
            lines.append(
                "| --- | --- | --- | --- | --- | --- |"
            )
            lines.append(
                f"| {entry['opened']} | {entry['last_updated']} | {entry['closed']} | "
                f"{entry['merged_into']} | {entry['merge_strategy']} | {entry['summary_link']} |"
            )
            lines.append("")
            lines.append("---")
            lines.append("")

    write_text(TIMELINE_PATH, "\n".join(lines).rstrip() + "\n")


def detect_merged_branch_from_head() -> Optional[str]:
    msg = run_git(["log", "-1", "--pretty=%B"])
    patterns = [
        r"Merge branch '([^']+)'",
        r"Merge pull request #[0-9]+ from [^/]+/(.+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, msg)
        if m:
            return m.group(1).strip()
    return None


def mark_branch_closed(branch: str, merged_into: str, strategy: str) -> None:
    if not branch or branch in MAIN_BRANCHES:
        return
    path = ensure_branch_summary(branch)
    content = read_text(path)
    content = update_metadata_value(content, "Status", "closed")
    content = update_metadata_value(content, "Closed on", today_str())
    content = update_metadata_value(content, "Merged into", merged_into or "-")
    content = update_metadata_value(content, "Merge strategy", strategy or "-")
    content = update_metadata_value(content, "Last updated", today_str())
    write_text(path, content)


def cmd_post_checkout(_: argparse.Namespace) -> int:
    branch = current_branch()
    if branch and branch not in MAIN_BRANCHES:
        ensure_branch_summary(branch)
    regenerate_timeline()
    return 0


def cmd_pre_commit(_: argparse.Namespace) -> int:
    branch = current_branch()
    if not branch:
        regenerate_timeline()
        return 0
    if branch in MAIN_BRANCHES:
        regenerate_timeline()
        return 0

    summary_path = ensure_branch_summary(branch)
    ok, errors, updated_content = validate_summary(summary_path, set_last_updated=True)
    write_text(summary_path, updated_content)
    regenerate_timeline()
    if ok:
        return 0

    print("Branch summary validation failed:")
    for err in errors:
        print(f"- {err}")
    print(f"Fix summary file and retry: {summary_path.relative_to(REPO_ROOT)}")
    return 1


def cmd_post_merge(args: argparse.Namespace) -> int:
    # post-merge gets one arg: squash(0|1). We still rely on HEAD message
    # and current branch to infer merged branch when possible.
    _ = args
    merged_branch = detect_merged_branch_from_head()
    target_branch = current_branch()
    if merged_branch and merged_branch != target_branch:
        strategy = "squash" if args.squash == "1" else "merge commit"
        mark_branch_closed(merged_branch, target_branch or "-", strategy)
    regenerate_timeline()
    return 0


def cmd_regenerate_timeline(_: argparse.Namespace) -> int:
    regenerate_timeline()
    return 0


def cmd_ensure_current(_: argparse.Namespace) -> int:
    branch = current_branch()
    if branch and branch not in MAIN_BRANCHES:
        ensure_branch_summary(branch)
    regenerate_timeline()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain branch summary docs and timeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    post_checkout = sub.add_parser("post-checkout")
    post_checkout.set_defaults(func=cmd_post_checkout)

    pre_commit = sub.add_parser("pre-commit")
    pre_commit.set_defaults(func=cmd_pre_commit)

    post_merge = sub.add_parser("post-merge")
    post_merge.add_argument("squash", nargs="?", default="0")
    post_merge.set_defaults(func=cmd_post_merge)

    timeline = sub.add_parser("regenerate-timeline")
    timeline.set_defaults(func=cmd_regenerate_timeline)

    ensure = sub.add_parser("ensure-current")
    ensure.set_defaults(func=cmd_ensure_current)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = args.func(args)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
