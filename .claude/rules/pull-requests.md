# Pull request rules

- Work on a branch and open a PR; do not push directly to `main`. Never force-push `main`.
- Keep a PR to one topic. Fill in `.github/pull_request_template.md`.
- Before opening: `pytest -q`, `ruff check`, `mypy` (the same as CI).
- User-visible change: add a line to `CHANGELOG.md`.
- Label every accuracy or speed claim as one of: proved (Lean / paper), certified at run time
  (`verify=True`), measured (say device and torch version), or empirical (calibrated, not proved).
  Do not state more than the paper (Zenodo 10.5281/zenodo.24182849) supports.
- Do not commit large raw logs; put them in a release asset or Zenodo.
- Do not merge or release without the maintainer's approval.
