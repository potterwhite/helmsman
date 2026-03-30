# AI-First Documentation System — Portable Template

> **Purpose:** Inject a structured, AI-optimised documentation system into any project
> so that a coding agent (Claude Code, Cursor, Copilot, etc.) orients itself instantly
> **without scanning source code**.
>
> Battle-tested in SynapseERP and HarborPilot.  Copy the structure; adapt the content.
> **Related:** [中文版 →](../../zh/1-for-ai/ai_docs_system_template.md)

---

## Why this system exists

When an AI agent starts a session it normally does one of two things:

| Without this system | With this system |
|---|---|
| Reads 50–200 source files to understand the project | Reads 3–4 doc files, then writes code |
| Spends 40–60 % of token budget just *understanding* | Spends > 90 % of tokens on the actual task |
| Invents patterns that conflict with existing ones | Follows the project's own patterns |
| Forgets conventions across sessions | Re-reads the same docs every session in seconds |

**The key insight: the AI does not need to read code to understand structure —
it needs a well-maintained human-readable map.**

Startup cost: ~2,000–4,000 tokens.
Savings per session: 20,000–80,000 tokens (eliminates source-code scanning).
ROI is positive after the very first task.

---

## The complete file set

```
<repo-root>/
├── CLAUDE.md                             ← Auto-injected every turn; session entry point
└── docs/
    └── architecture/
        ├── 00_INDEX.md                   ← Human + AI navigation hub (all docs in one table)
        ├── 1-for-ai/
        │   ├── guide.md                  ← Rules · commit format · workflow · architecture facts
        │   ├── codebase_map.md           ← File-by-file reference (replaces source scanning)
        │   └── <domain>.md              ← Deep-reference docs (API spec, parsing rules, etc.)
        ├── 2-progress/
        │   ├── progress.md              ← Phase status · completed steps · roadmap
        │   └── NEED_TO_DO.md           ← Living bug/task backlog (checkbox format)
        └── 3-highlights/
            ├── architecture_vision.md   ← Strategic positioning · design decisions
            └── archived/               ← Superseded decisions — kept for context, never deleted
```

**Read frequency:**

| File | When read | Purpose |
|---|---|---|
| `CLAUDE.md` | Every turn (auto-injected) | Entry point — must stay under 60 lines |
| `guide.md` | Once per session | Rules + facts |
| `codebase_map.md` | Once per session | Structure + file reference |
| `progress.md` | Once per session | Current phase context |
| `NEED_TO_DO.md` | When working on bugs/tasks | Active backlog |
| `00_INDEX.md` | When navigating | Master directory of all docs |
| Deep-reference docs | Only when the task needs it | API spec, domain rules, etc. |

---

## File 0: `CLAUDE.md` — Session Entry Point

**Auto-injected into every prompt by Claude Code.**  Keep it under 60 lines — all
detail goes into the doc files it points to.

The file has four sections:

1. **Session Start Protocol** — ordered reading list, explicit "do NOT scan src/"
2. **Requirements** — language, communication style
3. **Commands** — the project's key CLI commands (copy from README)
4. **Documentation Map** — one-line table mapping "need → file"

### Template

```markdown
# CLAUDE.md

This file provides essential guidance to Claude Code when working in this repository.
**For full context, always start by reading the docs** (see below).

## ⚠️ Session Start Protocol

**Before writing any code**, read these two files in order:

1. `docs/architecture/1-for-ai/guide.md` — working rules, commit format, key architecture facts
2. `docs/architecture/1-for-ai/codebase_map.md` — full codebase map (do NOT scan files instead)

Then check `docs/architecture/2-progress/progress.md` for current phase status.

Do **not** scan `<primary-source-dir>/` before reading the above.

---

## Requirements

- All source code and comments must be in **English**
- Communicate with me in **<preferred language>**
- Do not end the session arbitrarily — always prompt the user for next steps

---

## Commands

<paste the project's key CLI commands here>

---

## Documentation Map

| Need | File |
|---|---|
| Working rules + commit format | `docs/architecture/1-for-ai/guide.md` |
| Codebase structure | `docs/architecture/1-for-ai/codebase_map.md` |
| Phase progress + roadmap | `docs/architecture/2-progress/progress.md` |
| Active task backlog | `docs/architecture/2-progress/NEED_TO_DO.md` |
| Architecture vision | `docs/architecture/3-highlights/architecture_vision.md` |
```

> **Why name the specific source directory?**  `Do not scan src/` is too generic.
> Write the real path: `Do not scan backend/src/ or frontend/src/`.  The agent will
> respect a concrete instruction more reliably than a vague one.

---

## File 1: `docs/architecture/00_INDEX.md` — Navigation Hub

A lightweight index that lets any reader (human or AI) find any document in one
glance.  It is **not** a content file — it contains only tables that point elsewhere.

```markdown
# <ProjectName> — Documentation Index

> Last updated: <date> · Status: <current phase summary>

---

## 1 · For AI Agents & Developers (`1-for-ai/`)

Start here every session. Read in this order before touching any code.

| Document | Purpose |
|---|---|
| [`1-for-ai/guide.md`](1-for-ai/guide.md) | ⭐ Working rules, commit format, workflow, key architecture facts |
| [`1-for-ai/codebase_map.md`](1-for-ai/codebase_map.md) | ⭐ Full codebase structure — read this instead of scanning files |
| [`1-for-ai/<domain>.md`](1-for-ai/<domain>.md) | <purpose> |

---

## 2 · Project Progress (`2-progress/`)

| Document | Purpose |
|---|---|
| [`2-progress/progress.md`](2-progress/progress.md) | All Phase step details, current status, future roadmap |
| [`2-progress/NEED_TO_DO.md`](2-progress/NEED_TO_DO.md) | Active bug/task backlog (working notes) |

---

## 3 · Architecture & Highlights (`3-highlights/`)

| Document | Purpose |
|---|---|
| [`3-highlights/architecture_vision.md`](3-highlights/architecture_vision.md) | Strategic positioning and architecture philosophy |
| [`3-highlights/archived/`](3-highlights/archived/) | Historical decisions — kept for context, no longer active |
```

---

## File 2: `1-for-ai/guide.md` — Working Rules

Read every session, before codebase_map.  Contains:

1. **Reading Order** — where to go after this file
2. **Non-negotiable rules** — language, commits, documentation
3. **Commit message format** — exact format with examples
4. **How to handle request types** — new feature / bug / refactor
5. **Common pitfalls** — project-specific wrong patterns
6. **Key architecture facts** — 5–10 facts the agent must never violate
7. **Development commands** — mirror of CLAUDE.md commands (redundancy is intentional)

### Template

```markdown
# <ProjectName> — AI Agent Guide

> **Target audience:** AI coding agents (Claude Code, Cursor, etc.)
> **Read this before touching any code.**

---

## 1. Reading Order (every session)

1. **This file** — understand how to work in this repo
2. **[`codebase_map.md`](codebase_map.md)** — full codebase structure (replaces scanning)
3. **[`../2-progress/progress.md`](../2-progress/progress.md)** — current phase + active tasks
4. **Relevant reference doc** — only if your task requires it (API spec, domain rules, etc.)

---

## 2. Non-Negotiable Rules

### Code
- All source code and comments must be in **English**
- Communicate with the human in **<preferred language>**
- Do **not** end the session — always prompt the user for next steps

### Commits
- **One commit per STEP** — do not accumulate changes and commit at the end
- Follow the commit message format below exactly
- Never commit broken code or failing tests

### Documentation
- After modifying any file listed in `codebase_map.md`, update that file in the same commit
- When a Phase step is completed, update the status in `progress.md`

---

## 3. Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Type** (required): `feat` · `fix` · `docs` · `refactor` · `perf` · `test` · `build` · `chore`

**Subject** (required): English, ≤70 chars, present tense
- ✅ `fix: handle null value in config loader when PORT_SLOT is unset`
- ❌ `Fixed bugs and updated stuff`

**Body** (recommended): bullet points explaining what and why

**Footer** (recommended): `Phase X.Y Step Z complete.`

---

## 4. How to Handle Human Requests

### "Build a new feature"
1. Ask clarifying questions (goal, affected files, breaking changes)
2. Write a plan in `docs/` — **no code yet**
3. Wait for approval
4. Implement step by step, one commit per step

### "There's a bug"
1. Reproduce and understand the root cause
2. Fix it
3. Commit with `fix:` prefix

### "Refactor / optimize something"
1. Write a refactor plan in `docs/architecture/`
2. Wait for approval
3. Execute step by step

---

## 5. Common Pitfalls

| ❌ Wrong | ✅ Right |
|---|---|
| Edit 10 files then do one big commit | Commit after each logical step |
| Start coding without reading codebase_map | Read codebase_map first |
| Edit code, forget to update codebase_map | Always sync codebase_map in same commit |
| Vague commit message "fix bugs" | Specific message naming the root cause |
| Invent new architectural patterns | Follow existing patterns in codebase_map |

---

## 6. Key Architecture Facts

- **<Fact 1>** — e.g. "Config is the single source of truth; no hardcoded values in scripts"
- **<Fact 2>** — e.g. "Three-layer config: defaults/ → common.env → platform.env (last wins)"
- **<Fact 3>** — e.g. "Ports are never hardcoded; always derived from PORT_SLOT via port_calc.sh"
- *(5–10 facts — the things the agent most commonly gets wrong)*

---

## 7. Development Commands

```bash
<paste the same commands as CLAUDE.md — redundancy is intentional>
```
```

> **Why duplicate commands in guide.md?**  `CLAUDE.md` is read every turn (short,
> injected into context).  `guide.md` is read once per session as a full document.
> Duplicating the commands means the agent always has them in front of it, regardless
> of which file it is currently in.

---

## File 3: `1-for-ai/codebase_map.md` — The Most Important File

This replaces source-code scanning.  A well-written map lets the agent act on any
file without opening it.

### Required sections

1. **Warning header** + maintenance rule + Last updated (with reason)
2. **Repository root layout** (ASCII tree)
3. **File-by-file reference** for every non-trivial module
4. **Key architectural patterns** (numbered, 1–2 sentences each)

### Header (copy verbatim, fill in)

```markdown
# <ProjectName> — Codebase Map (AI Agent Quick Reference)

> **⚠️ FOR AI AGENTS — READ THIS FIRST**
> This document is the single source of truth for codebase structure.
> **Do NOT do a full repo scan** — read this file instead.
>
> **Maintenance rule:** Any AI agent that modifies a file listed here MUST update
> the relevant section in this document in the same commit/session.
>
> Last updated: <date> (<one-line reason, e.g. "added port_calc.sh section">)
```

> **Why put the maintenance rule in the codebase_map header?**  The agent reads
> codebase_map before every coding task.  If the rule is only in CLAUDE.md or
> guide.md, the agent may have forgotten it by the time it finishes coding.
> Placing it at the top of codebase_map is the last thing the agent reads before
> touching files.

### Good vs bad entry

**Bad** (too vague — forces the agent to open the file anyway):
```
### `scripts/port_calc.sh`
Port calculation script.
```

**Good** (enough detail to act without opening the file):
```
### `scripts/port_calc.sh`
Sourced after Layer 3 in every config loader.  Two mutually exclusive modes:
- MODE A (recommended): set `PORT_SLOT` in platform .env → all ports derived automatically
  - Formula: `CLIENT_SSH_PORT = 2109 + PORT_SLOT * 10`, `GDB_PORT = 2345 + PORT_SLOT * 10`
- MODE B (legacy): set `CLIENT_SSH_PORT` and `GDB_PORT` explicitly (no PORT_SLOT)
- Mixing modes → FATAL error with remediation instructions
- Cleans up internal `_*` variables after calculation
```

### "Last updated" convention

Always include the reason, not just the date:

```
Last updated: 2026-03-26 (added port_calc.sh; updated rk3588s to Ubuntu 24.04)
```

This lets the agent judge freshness without needing to open git log.

---

## File 4: `2-progress/progress.md` — Phase Status

Prevents the agent from proposing already-done features or contradicting completed
architecture decisions.

```markdown
# <ProjectName> — Progress

> Last updated: <date>

---

## Overall Status

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | <description> | ✅ Done |
| **Phase 2** | <description> | 🔄 In Progress |
| **Phase 3** | <description> | ⏳ Pending |

**Currently active:** Phase 2.3 — <active step name>

---

## Phase 1 — <Name>

| Step | Description | Commit |
|---|---|---|
| **1.1** | <what was done> | `abc1234` |
| **1.2** | <what was done> | `def5678` |

---

## Phase 2 — <Name> (In Progress)

| Item | Status |
|---|---|
| <item> | ✅ Done |
| <item> | 🔄 In progress |
| <item> | ⏳ Not started |
```

---

## File 5: `2-progress/NEED_TO_DO.md` — Living Backlog

A plain checkbox list.  The agent reads it, does unchecked items, checks them off.

### Format rules

```markdown
- **改完就把下面的checkbox checked**   ← Always keep this reminder line at top

<Month><Day>.<Year> <time>
- [x] <completed item>
    ```bash
    <error output if relevant>
    ```
- [ ] <pending item>
- [ ] <pending item>


<Earlier date group below>
<Month><Day>.<Year> <time>
- [x] <older completed item>
```

**Conventions:**
- Newest date group at the **top**
- Checked `[x]` = done — never delete them, they are history
- Inline code blocks for bug output so the agent understands context immediately
- One date group per working session (not per day if you have multiple sessions)

---

## Optional: Deep-Reference Documents (`1-for-ai/<domain>.md`)

For projects with complex domain rules that would clutter codebase_map, add separate
reference files.  Only create one when you have more than ~30 lines of domain-specific
rules that an agent needs to look up regularly.

| Example file | When to create it |
|---|---|
| `api_spec.md` | REST API with > 10 endpoints and non-obvious request/response shapes |
| `obsidian_parsing_rules.md` | Custom file-format parsing with many edge cases |
| `frontend_config.md` | Non-standard build config, env var conventions, proxy rules |
| `config_schema.md` | Config system with many fields, types, and interaction rules |

In `guide.md` Section 1 (Reading Order), add:
```
4. **Relevant reference doc** — only if your task requires it (api_spec.md, etc.)
```

In `00_INDEX.md`, add a row to the `1-for-ai/` table for each new file.

---

## The `3-highlights/archived/` convention

When an architectural decision is superseded, **never delete it**.  Move it to
`3-highlights/archived/` with its original filename.  Add a one-line note at the top:

```markdown
> **ARCHIVED** — superseded by Phase 5.2 DB-Primary decision (2026-03-25).
> Kept for context only.
```

This gives the agent (and new team members) the full history of *why* the current
architecture is the way it is.

---

## Injecting into a new project — checklist

```
[ ] CLAUDE.md at repo root
      - Session start protocol pointing to guide + codebase_map
      - Explicit "Do not scan <source-dir>/"
      - Key CLI commands
      - Documentation Map table
      - Keep under 60 lines

[ ] docs/architecture/00_INDEX.md
      - Three tables: 1-for-ai / 2-progress / 3-highlights
      - One row per file, with relative link and one-line purpose

[ ] docs/architecture/1-for-ai/guide.md
      - Reading Order section (with "only if needed" for deep-reference docs)
      - Non-negotiable rules (language + commits + docs maintenance)
      - Commit format with real examples from this project
      - Request handling: feature / bug / refactor
      - Common pitfalls (at least 5, project-specific)
      - Key architecture facts (5–10, the things agents get wrong most)
      - Dev commands (duplicate of CLAUDE.md — intentional)

[ ] docs/architecture/1-for-ai/codebase_map.md
      - Warning header with maintenance rule embedded at top
      - "Last updated: <date> (<reason>)" line
      - Full ASCII tree of repo root
      - Every non-trivial file: path + function + key variables/patterns
      - Architectural patterns section at the end

[ ] docs/architecture/2-progress/progress.md
      - Overall status table (all phases, emoji status)
      - "Currently active: Phase X.Y" line
      - Detail for each completed and in-progress phase

[ ] docs/architecture/2-progress/NEED_TO_DO.md
      - Reminder line at top
      - At least one date group
      - Checkbox format; newest at top

[ ] docs/architecture/3-highlights/architecture_vision.md
      - Why the project is built this way
      - Key design decisions with rationale

[ ] Add maintenance rule to BOTH CLAUDE.md AND guide.md AND codebase_map header:
      "Any agent that modifies a file in codebase_map must update codebase_map
       in the same commit."
```

---

## Common mistakes when setting up this system

| ❌ Mistake | ✅ Correct approach |
|---|---|
| CLAUDE.md over 100 lines | Keep under 60; move all detail to guide.md |
| codebase_map entries are one sentence: "Contains API views" | Include function names, key params, auth requirements |
| `Last updated` date with no reason | Always add reason: `(added X, updated Y to Z)` |
| Deleting superseded design docs | Move to `3-highlights/archived/` |
| Putting domain rules directly in codebase_map | Create a separate `1-for-ai/<domain>.md` when > 30 lines |
| Only adding maintenance rule to guide.md | Embed it in codebase_map header too — that's where agents read it last |
| Telling agent "don't scan src/" | Tell agent the exact path: "don't scan `backend/src/` or `scripts/`" |
| progress.md lists only future phases | Include completed phases with commit hashes — history prevents repetition |
