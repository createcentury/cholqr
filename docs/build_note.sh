#!/bin/bash
# Build the cholqr technical note: (1) MDX blog post with Google Scholar meta tags into the blog repo,
# (2) PDF (pandoc + lualatex, searchable text, title/author/date as the largest text) into docs/build/.
# Usage: docs/build_note.sh [BLOG_REPO]     (default ~/createcentury.github.io)
# Edit the metadata block below before publishing (date, slug/number, URLs).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BLOG="${1:-$HOME/createcentury.github.io}"
SRC="$HERE/rounding_note_DRAFT.md"

# ---- metadata (PLACEHOLDERS until publication) ----
TITLE="When does shifted CholeskyQR break in FP32? An a-posteriori bound, an FP64 certificate, and measurements across hardware"
DATE="2026/12/01"                                   # citation_publication_date, YYYY/MM/DD  [PLACEHOLDER]
NUMBER=9; SLUG=9                                    # next blog post number on createcentury.github.io
PAGE_URL="https://createcentury.github.io/blog/$SLUG"
PDF_NAME="cholqr-rounding-note.pdf"
PDF_URL="https://createcentury.github.io/blog/$PDF_NAME"   # same directory as /blog/<slug> (Scholar requirement)
POST="$BLOG/blog/${DATE//\//-}-cholqr-rounding-note.mdx"

mkdir -p "$HERE/build"
python3 "$HERE/note_to_mdx.py" "$SRC" "$POST" "$HERE/build/note_pandoc.md" \
  --title "$TITLE" --date "$DATE" --page-url "$PAGE_URL" --pdf-url "$PDF_URL" --slug "$SLUG" --number "$NUMBER"

pandoc "$HERE/build/note_pandoc.md" -o "$HERE/build/$PDF_NAME" --pdf-engine=lualatex \
  -V mainfont="STIX Two Text" -V mathfont="STIX Two Math" -V monofont="Menlo" \
  -V mainfontfallback="STIX Two Math:mode=harf" -V mainfontfallback="Menlo:mode=harf" \
  -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true -V urlcolor=blue -V linkcolor=blue \
  --highlight-style=tango
mkdir -p "$BLOG/static/blog"
cp "$HERE/build/$PDF_NAME" "$BLOG/static/blog/$PDF_NAME"        # served at /blog/$PDF_NAME
echo "PDF:  $HERE/build/$PDF_NAME ($(du -h "$HERE/build/$PDF_NAME" | cut -f1))"
echo "Post: $POST"
