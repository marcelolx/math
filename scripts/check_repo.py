"""Repository-wide consistency & verification gate for marcelolx/math.

Inspired by the publication validation scripts in 1337isnot1337/solved-conjectures.
Run from the repository root before committing or promoting to main:

    uv run python scripts/check_repo.py

Checks performed:
  1. Python stdlib safety: ensures no top-level `math.py` or `math/` package exists.
  2. Topic & entry completeness: every `<topic>/entries/*.md` file is linked in `<topic>/README.md`
     and every entry link in `<topic>/README.md` exists on disk.
  3. Entry metadata contract: every entry declares Date, Status, and a Verification Script path
     that resolves to an existing `.py` (or `.lean`) file.
  4. Relative Markdown links: checks local file links across root docs, `reviews/README.md`,
     track READMEs, `SYNTHESIS.md` files, and entry files.
  5. GitHub Pages assets: checks every `src="assets/..."` reference in `docs/index.html` exists.
  6. JSON integrity: validates every `results/*_metrics.json` parses cleanly with no NaN/Infinity.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TOPICS = ("rocket-combustion", "plasma-nozzles", "grid-stability")

MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
IMG_SRC_RE = re.compile(r'src="(assets/[^"]+)"')


def check_no_nan(obj, path_str: str, errors: list[str]) -> None:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            errors.append(f"JSON NaN/Inf in {path_str}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_no_nan(v, f"{path_str}.{k}", errors)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_no_nan(v, f"{path_str}[{i}]", errors)


def check_md_links(md_path: Path, errors: list[str]) -> None:
    text = md_path.read_text(encoding="utf-8")
    # Strip fenced code blocks to avoid false positives on code snippets
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    for raw_target in MD_LINK_RE.findall(cleaned):
        target = raw_target.strip()
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        file_part = target.split("#", 1)[0].strip()
        if not file_part:
            continue
        resolved = (md_path.parent / file_part).resolve()
        if not resolved.exists():
            rel = md_path.relative_to(ROOT)
            errors.append(f"Broken link in {rel}: ({target}) -> {resolved} not found")


def main() -> int:
    errors: list[str] = []
    stats = {"topics": 0, "entries": 0, "json_files": 0, "md_checked": 0}

    # 1. Stdlib math shadowing guard
    if (ROOT / "math.py").exists() or (ROOT / "math").is_dir():
        errors.append("Forbidden `math.py` or `math/` directory found at repo root (shadows stdlib `math`).")

    # 2. Required governance files
    for req in ("README.md", "AGENTS.md", "CITATION.cff", "HISTORY.md", "reviews/README.md", "templates/progress-block.md"):
        p = ROOT / req
        if not p.exists():
            errors.append(f"Missing required governance file: {req}")
        elif p.suffix == ".md":
            check_md_links(p, errors)
            stats["md_checked"] += 1

    # 3. Topic completeness, entry metadata, links, and JSON integrity
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for topic in TOPICS:
        tdir = ROOT / topic
        if not tdir.is_dir():
            errors.append(f"Missing topic directory: {topic}")
            continue
        stats["topics"] += 1

        t_readme = tdir / "README.md"
        t_synth = tdir / "SYNTHESIS.md"
        for doc in (t_readme, t_synth):
            if not doc.exists():
                errors.append(f"Missing {doc.relative_to(ROOT)}")
            else:
                check_md_links(doc, errors)
                stats["md_checked"] += 1

        t_readme_txt = t_readme.read_text(encoding="utf-8") if t_readme.exists() else ""
        entries_dir = tdir / "entries"
        entry_files = sorted(entries_dir.glob("*.md")) if entries_dir.is_dir() else []

        for ef in entry_files:
            stats["entries"] += 1
            rel_ef = ef.relative_to(ROOT)
            if ef.name not in t_readme_txt:
                errors.append(f"Unindexed entry {rel_ef}: not listed in {topic}/README.md")
            if f"{topic}/entries/{ef.name}" not in root_readme:
                errors.append(f"Unindexed entry {rel_ef}: not listed in root README.md catalogue")

            etxt = ef.read_text(encoding="utf-8")
            for field in ("- **Date:**", "- **Status:**", "- **Verification Script:**"):
                if field not in etxt:
                    errors.append(f"Missing metadata field `{field}` in {rel_ef}")

            m_vs = re.search(r"-\s+\*\*Verification Script:\*\*\s+`([^`]+)`", etxt)
            if m_vs:
                vs_path = (ef.parent / m_vs.group(1)).resolve()
                if not vs_path.exists():
                    errors.append(f"Missing verification script in {rel_ef}: {m_vs.group(1)}")
            check_md_links(ef, errors)
            stats["md_checked"] += 1

        results_dir = tdir / "results"
        if results_dir.is_dir():
            for jf in sorted(results_dir.glob("*_metrics.json")):
                stats["json_files"] += 1
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    check_no_nan(data, str(jf.relative_to(ROOT)), errors)
                except Exception as exc:
                    errors.append(f"Invalid JSON in {jf.relative_to(ROOT)}: {exc}")

    # 4. GitHub Pages docs/index.html asset checks
    docs_index = ROOT / "docs" / "index.html"
    if docs_index.exists():
        html = docs_index.read_text(encoding="utf-8")
        for asset in IMG_SRC_RE.findall(html):
            if not (ROOT / "docs" / asset).exists():
                errors.append(f"Missing GitHub Pages asset referenced in docs/index.html: docs/{asset}")

    if errors:
        print(f"FAIL: {len(errors)} repository consistency error(s) found:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(
        f"PASS: Checked {stats['topics']} tracks, {stats['entries']} entries, "
        f"{stats['md_checked']} Markdown docs, {stats['json_files']} JSON metric files — 0 errors."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
