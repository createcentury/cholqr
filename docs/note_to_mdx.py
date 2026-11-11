"""Convert docs/rounding_note_DRAFT.md into (a) a Docusaurus MDX blog post with Google Scholar (Highwire) meta tags and
(b) a pandoc-ready Markdown file for the PDF. Repeatable: called by docs/build_note.sh.

Usage: python3 note_to_mdx.py SRC.md OUT_POST.mdx OUT_PANDOC.md --title T --date YYYY/MM/DD --page-url URL --pdf-url URL
       [--slug N] [--number N]
"""
import argparse, re

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("post"); ap.add_argument("pandoc_md")
ap.add_argument("--title", required=True); ap.add_argument("--date", required=True)        # YYYY/MM/DD (Scholar format)
ap.add_argument("--page-url", required=True); ap.add_argument("--pdf-url", required=True)
ap.add_argument("--slug", default="9"); ap.add_argument("--number", default="9")
ap.add_argument("--author", default="Fukuda, Soki"); ap.add_argument("--author-display", default="Soki Fukuda")
a = ap.parse_args()

src = open(a.src, encoding="utf-8").read()
lines = src.split("\n")
# drop the H1 title and the "**Author** — DRAFT ..." byline: both come from metadata
while lines and (lines[0].startswith("# ") or lines[0].startswith("**Soki Fukuda**") or not lines[0].strip()):
    lines.pop(0)
body = "\n".join(lines)
body = body.replace("url    = {TBD}", "url    = {" + a.page_url + "}")
iso = a.date.replace("/", "-")


# ---------- (b) pandoc Markdown (HTML comments are dropped by the LaTeX writer) ----------
def yq(s): return '"' + s.replace('"', '\\"') + '"'
pandoc_md = ("---\ntitle: " + yq(a.title) + "\nauthor: " + yq(a.author_display) + "\ndate: " + yq(iso) +
             "\n---\n\n" + body + "\n")
open(a.pandoc_md, "w", encoding="utf-8").write(pandoc_md)


# ---------- (a) MDX: escape JSX-significant characters outside code ----------
def esc_inline(seg):
    return seg.replace("{", "\\{").replace("}", "\\}").replace("<", "&lt;")


def esc_line(line):
    parts = re.split(r"(`[^`]*`)", line)            # keep inline code spans verbatim
    return "".join(p if p.startswith("`") and p.endswith("`") and len(p) > 1 else esc_inline(p) for p in parts)


# HTML comments are internal source notes (they point to private files): drop them from the public post
body_pub = re.sub(r"<!--.*?-->\n?", "", body, flags=re.S)
out, in_code, in_comment = [], False, False
for line in body_pub.split("\n"):
    if line.strip().startswith("```"):
        in_code = not in_code; out.append(line); continue
    if in_code:
        out.append(line); continue
    # HTML comments -> MDX comments (single- or multi-line)
    if in_comment or "<!--" in line:
        line = line.replace("<!--", "{/*").replace("-->", "*/}")
        if in_comment and "*/}" not in line:
            out.append(line); continue
        if "{/*" in line and "*/}" not in line:
            in_comment = True; out.append(line); continue
        in_comment = False
        pre, _, rest = line.partition("{/*")
        cmt, _, post = rest.partition("*/}")
        out.append(esc_line(pre) + ("{/*" + cmt + "*/}" if _ else cmt) + esc_line(post)); continue
    out.append(esc_line(line))
mdx = "\n".join(out)
# truncate marker after the abstract (first "## " heading after "## Abstract")
idx = [m.start() for m in re.finditer(r"^## ", mdx, flags=re.M)]
if len(idx) >= 2:
    mdx = mdx[: idx[1]] + "{/* truncate */}\n\n" + mdx[idx[1]:]

head = f"""---
title: "#{a.number} {a.title}"
slug: {a.slug}
date: {iso}T09:00:00+09:00
authors: [createcentury]
tags: [ml]
draft: true
---

<head>
  <meta name="citation_title" content="{a.title}" />
  <meta name="citation_author" content="{a.author}" />
  <meta name="citation_publication_date" content="{a.date}" />
  <meta name="citation_technical_report_institution" content="createcentury.github.io" />
  <meta name="citation_pdf_url" content="{a.pdf_url}" />
  <meta name="citation_public_url" content="{a.page_url}" />
  <meta name="description" content="When shifted CholeskyQR3 breaks in FP32: a proven per-pass bound, a run-time test with fallback, and pre-registered measurements across hardware (cholqr technical note)." />
</head>

**{a.author_display}** · PDF: [{a.pdf_url.rsplit('/', 1)[-1]}]({a.pdf_url}) · Code: [createcentury/cholqr](https://github.com/createcentury/cholqr)

"""
footer = f"\n\n---\n\n*Created: {iso} / Updated: {iso}*\n"
open(a.post, "w", encoding="utf-8").write(head + mdx + footer)
print("wrote", a.post, "and", a.pandoc_md)
