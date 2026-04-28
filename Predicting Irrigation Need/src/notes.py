from __future__ import annotations

import argparse
from datetime import date

from benchmark import PROJECT_ROOT

IDEAS_PATH = PROJECT_ROOT / "artifacts" / "ideas.md"
DIARY_PATH = PROJECT_ROOT / "DIARY.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update workflow notes for future agents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_idea = subparsers.add_parser("add-idea", help="Append a new idea entry.")
    add_idea.add_argument("title")
    add_idea.add_argument("hypothesis")
    add_idea.add_argument("expected_signal")

    start_idea = subparsers.add_parser("start-idea", help="Mark an idea as tried.")
    start_idea.add_argument("title")

    note_idea = subparsers.add_parser("note-idea", help="Append an outcome note to an idea.")
    note_idea.add_argument("title")
    note_idea.add_argument("note")

    add_diary = subparsers.add_parser("add-diary", help="Append a dated diary note.")
    add_diary.add_argument("change")
    add_diary.add_argument("result")
    add_diary.add_argument("decision")
    add_diary.add_argument("follow_up")
    add_diary.add_argument("--date", dest="entry_date", default=date.today().isoformat())

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "add-idea":
        add_idea(args.title, args.hypothesis, args.expected_signal)
        return
    if args.command == "start-idea":
        update_idea(args.title, status="tried")
        return
    if args.command == "note-idea":
        update_idea(args.title, note=args.note)
        return
    if args.command == "add-diary":
        add_diary(args.entry_date, args.change, args.result, args.decision, args.follow_up)
        return
    raise ValueError(f"Unknown command: {args.command}")


def add_idea(title: str, hypothesis: str, expected_signal: str) -> None:
    ensure_ideas_file()
    entry = format_idea_entry("new", title, hypothesis, expected_signal, "pending")
    with IDEAS_PATH.open("a", encoding="utf-8") as handle:
        if not IDEAS_PATH.read_text(encoding="utf-8").endswith("\n"):
            handle.write("\n")
        handle.write(f"{entry}\n")
    print(f"Added idea: {title}")


def update_idea(title: str, status: str | None = None, note: str | None = None) -> None:
    ensure_ideas_file()
    lines = IDEAS_PATH.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    matched = False

    for line in lines:
        if is_idea_line_for_title(line, title) and not matched:
            matched = True
            updated_lines.append(rewrite_idea_line(line, status=status, note=note))
        else:
            updated_lines.append(line)

    if not matched:
        raise ValueError(f"Idea not found: {title}")

    IDEAS_PATH.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    print(f"Updated idea: {title}")


def add_diary(entry_date: str, change: str, result: str, decision: str, follow_up: str) -> None:
    ensure_diary_file()
    content = DIARY_PATH.read_text(encoding="utf-8")
    lines: list[str] = []
    if content and not content.endswith("\n"):
        content += "\n"
    if content.strip():
        lines.append(content.rstrip("\n"))
    lines.extend(
        [
            f"## {entry_date}",
            "",
            f"- What changed and why: {change}",
            f"- Result: {result}",
            f"- Decision: {decision}",
            f"- Follow-up idea or open question: {follow_up}",
        ]
    )
    DIARY_PATH.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"Added diary entry for {entry_date}")


def ensure_ideas_file() -> None:
    if IDEAS_PATH.exists():
        return
    IDEAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDEAS_PATH.write_text(
        "# Ideas for Next Experiments\n\n"
        "Agents: append your ideas here before coding. Mark them `[tried]` when execution starts and add a short outcome note after the run.\n\n"
        "Suggested entry format:\n\n"
        "- `[new]` Idea title | hypothesis: ... | expect: ... | note: pending\n"
        "- `[tried]` Idea title | hypothesis: ... | expect: ... | note: improved smoke by ..., crashed because ..., or no signal\n",
        encoding="utf-8",
    )


def ensure_diary_file() -> None:
    if DIARY_PATH.exists():
        return
    DIARY_PATH.write_text(
        "# Diary\n\n"
        "Use this file for short dated research notes that help the next agent continue from the current frontier.\n\n"
        "Suggested format:\n\n"
        "## YYYY-MM-DD\n\n"
        "- What changed and why.\n"
        "- Result: did the benchmark improve? Include the run id and score when possible.\n"
        "- Decision: keep, discard, or revisit later.\n"
        "- Follow-up idea or open question.\n",
        encoding="utf-8",
    )


def format_idea_entry(status: str, title: str, hypothesis: str, expected_signal: str, note: str) -> str:
    return f"- `[{status}]` {title} | hypothesis: {hypothesis} | expect: {expected_signal} | note: {note}"


def is_idea_line_for_title(line: str, title: str) -> bool:
    return line.startswith("- `[") and f"]` {title} |" in line


def rewrite_idea_line(line: str, status: str | None = None, note: str | None = None) -> str:
    parts = line.split(" | ")
    head = parts[0]
    if status is not None:
        marker_start = head.find("`[")
        marker_end = head.find("]`", marker_start)
        if marker_start == -1 or marker_end == -1:
            raise ValueError(f"Idea line has unexpected format: {line}")
        head = f"{head[:marker_start]}`[{status}]`{head[marker_end + 2:]}"
    if note is not None:
        replaced = False
        for index, part in enumerate(parts):
            if part.startswith("note: "):
                parts[index] = f"note: {note}"
                replaced = True
                break
        if not replaced:
            parts.append(f"note: {note}")
        parts[0] = head
        return " | ".join(parts)
    parts[0] = head
    return " | ".join(parts)


if __name__ == "__main__":
    main()
