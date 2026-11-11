"""Convert docs/rounding_note_DRAFT.md into (a) a Hugo page-bundle post (index.md with Google Scholar params in the front
matter; the site's extend_head partial emits the Highwire meta tags) and (b) pandoc Markdown for the PDF.
Called by docs/build_note_hugo.sh.

Usage: python3 note_to_hugo.py SRC.md OUT_INDEX.md OUT_PANDOC.md --title T --date YYYY/MM/DD --page-url URL --pdf-name F
"""
import argparse, re

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("post"); ap.add_argument("pandoc_md")
ap.add_argument("--title", required=True); ap.add_argument("--date", required=True)      # YYYY/MM/DD (Scholar format)
ap.add_argument("--page-url", required=True); ap.add_argument("--pdf-name", required=True)
ap.add_argument("--author", default="Fukuda, Soki"); ap.add_argument("--author-display", default="Soki Fukuda")
ap.add_argument("--code-url", default="https://github.com/createcentury/cholqr")
ap.add_argument("--bibkey", default="fukuda2026cholqr")
ap.add_argument("--draft", default="true")
a = ap.parse_args()

lines = open(a.src, encoding="utf-8").read().split("\n")
while lines and (lines[0].startswith("# ") or lines[0].startswith("**Soki Fukuda**") or not lines[0].strip()):
    lines.pop(0)                                    # H1 title and byline come from metadata
body = "\n".join(lines)
body = re.sub(r"@misc\{[^,]+,", "@misc{" + a.bibkey + ",", body, count=1)
body = body.replace("url    = {TBD}", "url    = {" + a.page_url + "}")
body = re.sub(r"<!--.*?-->\n?", "", body, flags=re.S)  # internal source notes (private file pointers): never published
iso = a.date.replace("/", "-")


def yq(s): return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# (b) pandoc Markdown for the PDF
open(a.pandoc_md, "w", encoding="utf-8").write(
    "---\ntitle: " + yq(a.title) + "\nauthor: " + yq(a.author_display) + "\ndate: " + yq(iso) + "\n---\n\n" + body + "\n")

# (a) Hugo post
front = "\n".join([
    "---",
    "title: " + yq(a.title),
    "date: " + iso + "T09:00:00+09:00",
    "draft: " + a.draft,
    "math: true",
    "ShowToc: true",
    "citation_title: " + yq(a.title),
    "citation_author: [" + yq(a.author) + "]",
    "citation_publication_date: " + yq(a.date),
    "citation_pdf_url: " + yq("posts/" + a.page_url.rstrip("/").split("/")[-1] + "/" + a.pdf_name),
    "---", ""])
byline = (f"*{a.author_display}* · [PDF]({a.pdf_name}) · [Code]({a.code_url})\n\n")
open(a.post, "w", encoding="utf-8").write(front + byline + body.strip() + "\n")
