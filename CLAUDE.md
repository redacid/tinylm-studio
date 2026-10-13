# CLAUDE.md

All agent instructions for this project live in [AGENTS.md](./AGENTS.md). Read it before starting any work.

Key rules (see AGENTS.md for details):
- Talk to the user in Ukrainian or Russian, matching the language they write in.
- All code, code comments, commit messages, PR descriptions and project documentation are written in English.
- The engine runs in a container in a Kubernetes cluster: keep it configured by `TINYLM_*` variables with all state under `TINYLM_DATA_DIR` (see docs/deployment.md).
- Every new or changed feature, setting, `make` target or API endpoint updates the docs (README, docs/user-guide.md, docs/api.md, .env.example) in the same commit, with screenshots for visible changes.
