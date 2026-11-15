---
name: open-pr
description: Open a pull request for the current branch of cholqr following the repo's PR rules.
---

1. Read `.claude/rules/pull-requests.md`.
2. Run `pytest -q`, `ruff check`, `mypy`; fix failures or report them.
3. Check `CHANGELOG.md` is updated for user-visible changes.
4. Push the branch, then `gh pr create` with a body filled from `.github/pull_request_template.md`.
5. Do not merge. Report the PR URL and anything unchecked.
