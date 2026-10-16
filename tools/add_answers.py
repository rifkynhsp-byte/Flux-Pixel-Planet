#!/usr/bin/env python3
"""Insert click-to-reveal answer blocks into chapters.

  python tools/add_answers.py answers/batch_01.json

The JSON maps a chapter stem to {"en": ["answer 1", ...], "id": ["jawaban 1", ...]}.
Each language gets a collapsed callout placed at the end of its exercise section
("## Exercises" / "## Latihan"), the same pattern the book already uses:

    ::: {.callout-tip collapse="true"}
    ## Answers
    **1.** ...
    :::

Chapters that already have a collapsed answer block are left unchanged.
"""
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEAD = {"en": ("Exercises", "Answers"), "id": ("Latihan", "Jawaban")}


def insert(path, lang, answers):
    s = path.read_text(encoding="utf-8")
    if 'collapse="true"' in s:
        return "skip (has answers)"
    ex, title = HEAD[lang]
    m = re.search(rf"^## {ex}\b.*$", s, flags=re.M)
    if not m:
        return "skip (no exercise section)"
    nxt = re.search(r"^## ", s[m.end():], flags=re.M)
    end = m.end() + nxt.start() if nxt else len(s)
    n_ex = len(re.findall(r"^\d+\.\s", s[m.end():end], flags=re.M))
    if n_ex != len(answers):
        return f"MISMATCH ({n_ex} exercises, {len(answers)} answers)"
    block = "\n::: {.callout-tip collapse=\"true\"}\n## " + title + "\n\n" + \
            "\n\n".join(f"**{i}.**" + ("\n\n" if a.startswith("```") else " ") + a for i, a in enumerate(answers, 1)) + "\n:::\n"
    body = s[:end].rstrip("\n") + "\n" + block + ("\n" + s[end:] if nxt else "")
    path.write_text(body, encoding="utf-8")
    return f"ok ({n_ex})"


def main():
    data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    for stem, langs in data.items():
        for lang, answers in langs.items():
            print(f"{lang}/{stem}: {insert(ROOT / lang / f'{stem}.qmd', lang, answers)}")


if __name__ == "__main__":
    main()
