# Agent Guidelines

This file defines the rules every AI agent (Claude and any other assistant) must follow when working on this project.

## Language rules

### Communication with the user
- Talk to the user (Serhii) in **Ukrainian or Russian**.
- Match the language the user writes in: reply in Ukrainian to a Ukrainian message, in Russian to a Russian one.
- If the language is unclear, default to Ukrainian.

### Everything that goes into the project
Write the following **in English only**:
- Source code: identifiers, variable, function, class and file names.
- Code comments and docstrings.
- Commit messages and branch names.
- Pull request titles and descriptions, issue texts and review comments.
- Project documentation: README files, design notes, specs, this file and any other docs.
- Log messages, error messages and configuration files.

Do not mix languages inside code or project documentation. If the user dictates text in Ukrainian or Russian that must end up in the project, translate it into English before committing it.

## Documentation

Documentation is part of every change, not a follow-up:

- A new or changed feature, screen, setting (`TINYLM_*` variable), `make` target, CLI command or
  API endpoint is documented **in the same commit**:
  - [README.md](./README.md): what the project does and the quick start;
  - [docs/user-guide.md](./docs/user-guide.md): how to use each screen and setting, including
    setup of external pieces such as a proxy, a CDP browser or SearXNG;
  - [docs/api.md](./docs/api.md): endpoints, request fields and the settings table;
  - [.env.example](./.env.example): every setting a user is likely to change;
  - [docs/implementation-plan.md](./docs/implementation-plan.md) and [TODO.md](./TODO.md): what is
    done and what is left.
  - [docs/releasing.md](./docs/releasing.md): releases, how builds are named and the GitHub
    Actions workflows (`.github/workflows/`).
- When a screen is added or changes visibly, add or retake its screenshot in `docs/images/` and
  show it in the user guide.
- A new or changed endpoint also keeps the OpenAPI schema right: its response model in
  `engine/src/engine/api/schemas.py`, and the desktop app's generated types (`make api-types`,
  committed with the change); see [docs/api.md](./docs/api.md#openapi-schema).
- Before finishing a task, check that nothing the user can now do or configure is missing from
  the docs.

## Deployment target

The engine runs in a container in a Kubernetes cluster (decided 2026-10-13); the desktop app is
one of its clients. Keep every engine change working there, as described in
[docs/deployment.md](./docs/deployment.md):

- configuration only through `TINYLM_*` environment variables; no config files, no prompts;
- all state under `TINYLM_DATA_DIR`; nothing in the user's home folder, `/tmp` or the repository;
- no assumptions about the user's machine: no local paths, browsers or services on `127.0.0.1`
  that the engine needs without a setting to point elsewhere;
- one engine process per data folder; background jobs survive only as long as the pod;
- update docs/deployment.md when a change adds a setting, a volume path, outgoing network
  access or a new long-running job.

## Releases and versions

The version number is written in several places; change it only with `make release VERSION=x.y.z`
(see [docs/releasing.md](./docs/releasing.md)), never by hand, and never edit released sections
of CHANGELOG.md. Keep CI green: `make check` and `make version-check` before committing, and
`make api-types` after changing an endpoint (CI fails when the committed types are stale).

## Scope
These rules apply to every agent, thread and session in the project, and to any sub-agents or workers they start. Pass them on when briefing another agent.
