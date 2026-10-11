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

## Scope
These rules apply to every agent, thread and session in the project, and to any sub-agents or workers they start. Pass them on when briefing another agent.
