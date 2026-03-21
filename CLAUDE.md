# posefx-studio — Claude Code Instructions

Read and follow all rules in `AGENTS.md` — that is the project source of truth.
Read `.architecture.md` for current implementation status before making changes.
Read `BUILDPLAN.md` for the step-by-step build plan when asked to execute steps.

## Claude-specific notes
- When creating new files, follow the patterns in existing implemented files
- Run checkpoints after each build step to verify before continuing
- If a step fails, fix it before moving to the next step
- Prefer modifying existing files over creating parallel implementations