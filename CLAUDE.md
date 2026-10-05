# anchorite development notes

## Working norms

Operating directives for Claude (and any agent) in this repo; they counteract default model dispositions.

- **Resist the minimal-diff reflex.** Don't reach for the smallest change that hides the symptom (special-casing,
  papering over root causes). Aim for the correct fix at the right complexity level — not the smallest, not gold-plated.
- **Fail loudly and early.** Raise on a missing expected input or precondition; never fall back to a default/placeholder
  to limp along. A placeholder is an explicit caller input, never a code default.
- **Never instruct around a defect — fix the defect.** Don't write prose telling readers to work around broken code.
  Prose is untested, and callers who didn't read it stay broken.
- **Push back; don't just comply.** When a design, name, or approach seems worse — including a shortcut you're asked to
  take — say so with reasoning, unprompted. The author owns the final call.
- **Offer better alternatives with trade-offs.** When a materially better approach than the proposed one exists, present
  it and the trade-offs — don't just execute the ask.
- **Investigate before producing.** Read the code and verify constraints first. Don't treat a training-pattern
  convention as load-bearing unchecked; don't speculate about what you can read.
- **Explain non-obvious changes first.** For a change whose rationale isn't self-evident, give the why before showing or
  applying the diff.
- **Ask when unsure** rather than assume intent.
- **No intensifiers or emphasis filler.** Drop words and phrases that add emphasis but no information — "that's the
  key", "crucially", "importantly", "the key insight", "it's worth noting". State the point plainly. Applies to all
  prose: chat replies, PR/review comments, commit messages, and docs.

## Committing

- **Stage explicit paths**, not `git add -A` / `.`; explicit staging avoids sweeping in an untracked file. Scope
  repo-wide tools (`ruff check --fix`, formatters) to tracked paths for the same reason.
- **Pre-commit runs lint/format/type-check/hygiene** (`.pre-commit-config.yaml`); CI runs the same hooks plus pytest.
  Ensure hooks are installed (`pre-commit install`) — if not, install or ask the author; never bypass with
  `--no-verify`.
- **Correct a pushed branch with a new commit on top**, not amend + force-push. PRs squash-merge, so `main` history
  stays linear regardless and intermediate fixups vanish on merge. Reserve force-push for rebasing a branch onto `main`.

## Releasing

Bump `version` in `pyproject.toml` (and the anchorite entry in `uv.lock`) in a PR. After it merges, push a `v<version>`
tag on `main`; the release workflow checks the tag against the version, runs the tests, publishes to PyPI and creates
the GitHub release. Don't create the GitHub release by hand.

## Worktrees

Worktrees go in `.claude/worktrees/` (gitignored), never `../` siblings.
