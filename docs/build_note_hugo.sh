#!/bin/bash
# Build the cholqr technical note for the Hugo research site: page bundle content/posts/cholqr-rounding/
# (index.md with Google Scholar params + the PDF in the SAME directory, as Scholar requires).
# Usage: docs/build_note_hugo.sh [SITE_REPO]     (default ~/work/research-site)
# Edit the metadata block before publishing (date, draft flag).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SITE="${1:-$HOME/work/research-site}"
SRC="$HERE/rounding_note_DRAFT.md"

# ---- metadata (PLACEHOLDERS until publication) ----
TITLE="When does shifted CholeskyQR break in FP32? A proven bound, a run-time test, and measurements across hardware"
DATE="2026/12/01"                                   # citation_publication_date, YYYY/MM/DD  [PLACEHOLDER]
SLUG="cholqr-rounding"
PAGE_URL="https://createcentury.github.io/posts/$SLUG/"
PDF_NAME="$SLUG.pdf"
DRAFT="true"                                        # set to false to publish
BUNDLE="$SITE/content/posts/$SLUG"

mkdir -p "$HERE/build" "$BUNDLE"
python3 "$HERE/note_to_hugo.py" "$SRC" "$BUNDLE/index.md" "$HERE/build/note_pandoc_hugo.md" \
  --title "$TITLE" --date "$DATE" --page-url "$PAGE_URL" --pdf-name "$PDF_NAME" --draft "$DRAFT"
pandoc "$HERE/build/note_pandoc_hugo.md" -o "$HERE/build/$PDF_NAME" --pdf-engine=lualatex \
  -V mainfont="STIX Two Text" -V mathfont="STIX Two Math" -V monofont="Menlo" \
  -V mainfontfallback="STIX Two Math:mode=harf" -V mainfontfallback="Menlo:mode=harf" \
  -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true -V urlcolor=blue -V linkcolor=blue \
  --highlight-style=tango
cp "$HERE/build/$PDF_NAME" "$BUNDLE/$PDF_NAME"      # served at /posts/$SLUG/$PDF_NAME
echo "PDF:  $BUNDLE/$PDF_NAME ($(du -h "$BUNDLE/$PDF_NAME" | cut -f1))"
echo "Post: $BUNDLE/index.md"
