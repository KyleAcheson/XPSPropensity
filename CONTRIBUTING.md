# Contributing to XPSPropensity

How to develop the code. For what the program does, how to install and run it,
the package structure and how to add a new electronic-structure program, see
[README.MD](README.MD).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest                      # the whole suite must pass before any change is merged
```

## Where things are documented

| File | Contents |
|---|---|
| `docs/shakedown_cases.tex`, `docs/chp.tex` | The theory: shake-down cases, propensity rules and channels; the core-hole potential (CHP) |
| `CONTEXT.md` | Glossary: what each term means in this project, and which words to avoid |
| `docs/adr/` | Architecture decision records: decisions that were hard to make and should not be undone by accident |
| `AGENTS.md` | Code map, extension points, design principles and **coding standards** (for humans as well as agents) |
| Module docstrings | The detailed design of each module |

Read the ADRs that touch the area you are changing. If a change contradicts one,
say so in the issue or PR and add a new ADR rather than silently overriding it.

## Conventions

- Never commit to `main`. Work on a branch from `main` (`feature/<name>`) and
  merge through a pull request.
- No AI co-author lines (`Co-Authored-By: Claude ...`) in commit messages.
- Follow the coding standards in `AGENTS.md` and the vocabulary in `CONTEXT.md`.
- New behaviour comes with tests, checked against the data in `data/molcas`
  where possible.

## GitHub issues

Work is tracked as issues on GitHub. There are two kinds:

- **Spec**: one issue per feature describing the problem, the solution, user
  stories and the decisions made.
- **Ticket**: one issue per small, end-to-end piece of a spec. Each links to its
  spec and lists the tickets that block it (GitHub's "blocked by"
  dependencies, shown on the issue page). A ticket can be started once all its
  blockers are closed.

Labels:

| Label | Meaning |
|---|---|
| `ready-for-agent` | Fully specified; can be implemented (by an agent or a person) |
| `ready-for-human` | Needs a person to implement it |
| `needs-triage` | Not yet evaluated |
| `needs-info` | Waiting for more information from the reporter |
| `wontfix` | Will not be done |

A bug or request you find along the way is raised as a normal issue; it is
turned into a ready ticket by triage (`/triage` with the agent workflow below).

Issues are closed by the pull request that completes them: put
`Closes #13, closes #14` in the PR description.

## Development with Claude Code

Optional, but the repo is set up for it. You need
[Claude Code](https://claude.com/claude-code) with the **Skills For Real
Engineers** plugin (install with `/plugin`), and the GitHub CLI logged in
(`gh auth login`). The plugin's configuration for this repo is in
`docs/agents/`.

### A feature: grill, spec, tickets, implement

1. **Branch** from an up-to-date `main`:
   ```bash
   git switch main && git pull && git switch -c feature/<name>
   ```
2. **Grill** the idea in a fresh session: `/grill-with-docs <idea>`. The agent
   asks questions one round at a time, looks up facts in the code, the theory
   notes and the program sources, and you make the decisions. Terms are added
   to `CONTEXT.md` and decisions to `docs/adr/` as they are settled.
3. **Spec**, in the same session: `/to-spec`. The agent first proposes where
   the feature will be tested, for you to confirm, then posts the spec as one
   GitHub issue.
4. **Tickets**, in the same session: `/to-tickets`. Agree the list of tickets
   and their blockers, then the agent posts one issue per ticket.

   Do not `/clear` or `/compact` between steps 2 and 4: the spec and tickets
   build on the grilling.
5. **Implement** one ticket per context: `/clear`, then `/implement #13`. The
   agent builds the ticket test-first, runs the test suite, reviews the diff
   against `AGENTS.md` and the ticket (`/code-review`), and commits to the
   current branch (say "stop before committing" if you want to commit
   yourself). Then `/clear` and the next ticket whose blockers are done.
6. **Pull request** from the feature branch with `Closes #...` for the spec
   and its tickets; merging closes them.

### Small changes

For a change that fits in one session, skip the spec and tickets: grill if
needed, then `/implement` with a description of the change, on a feature
branch as usual.

### Other commands

- `/diagnosing-bugs`: for a bug that resists a first look; it first builds a
  command that reproduces the bug, then fixes it with a regression test.
- `/tdd`: build one behaviour test-first without a spec.
- `/code-review`: review a branch or PR against `AGENTS.md` and its issue.
- `/grill-me`: the same interview without updating `CONTEXT.md` or the ADRs.
- `/ask-matt`: which command to use when.
