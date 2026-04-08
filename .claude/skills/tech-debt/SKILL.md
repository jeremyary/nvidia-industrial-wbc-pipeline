---
description: Scan all consolidated review files for items with Decided=Defer and maintain a technical debt tracker at .changes/TECH_DEBT.md. Creates or updates the file each time it's run.
user_invocable: true
---

# Technical Debt Tracker

Scan all consolidated review files (current staging area and archived changesets) for items marked as `Defer` in the `Decided` column, and produce or update a single technical debt tracker at `.changes/TECH_DEBT.md`.

## Process

### 1. Find All Consolidated Reviews

Search for consolidated review files in both locations:
- `.changes/reviews/*-consolidated.md` (current staging area)
- `.changes/*/reviews/*-consolidated.md` (archived changesets)

### 2. Extract Deferred Items

For each consolidated review, parse every table row where the `Decided` column contains `Defer`. Extract:
- The finding ID (e.g., C-3, W-5, S-2)
- The finding description
- The original severity (Critical, Warning, Suggestion — inferred from the ID prefix or section heading)
- The suggested resolution
- The rationale from the Suggested Disposition column (the "why" behind the deferral)
- The source file path (which consolidated review it came from)

Also scan for items where `Decided` is still `_pending_` — list these separately as untriaged.

### 3. Check for Resolved Items

If `.changes/TECH_DEBT.md` already exists, read it and identify any items previously tracked that are no longer present in any consolidated review (the finding was removed or the review was deleted). Mark these as potentially resolved — but do NOT remove them automatically. Flag them for the user to confirm.

### 4. Write .changes/TECH_DEBT.md

Create or overwrite the file using this format:

```markdown
# Technical Debt

> Last updated: YYYY-MM-DD HH:MM
> Sources: N consolidated reviews scanned

## Active Debt

Items deferred during review triage. Each links back to its source review for full context.

### Security

| # | Finding | Original Severity | Suggested Resolution | Deferred Because | Source |
|---|---------|------------------|---------------------|-----------------|--------|
| TD-1 | <description> | Critical | <resolution> | <rationale> | <archive slug or "staging"> |

### Reliability

| # | Finding | Original Severity | Suggested Resolution | Deferred Because | Source |
|---|---------|------------------|---------------------|-----------------|--------|

### Code Quality

| # | Finding | Original Severity | Suggested Resolution | Deferred Because | Source |
|---|---------|------------------|---------------------|-----------------|--------|

### Ops & Deployment

| # | Finding | Original Severity | Suggested Resolution | Deferred Because | Source |
|---|---------|------------------|---------------------|-----------------|--------|

## Untriaged

Items where `Decided` is still `_pending_` — these need triage before they can be tracked or dismissed.

| # | Finding | Severity | Source |
|---|---------|----------|--------|

## Potentially Resolved

Items previously tracked here that no longer appear in any consolidated review. Confirm resolution before removing.

| # | Finding | Last Seen In | Notes |
|---|---------|-------------|-------|
```

### Categorization Rules

Assign each deferred item to a category based on its content:

- **Security** — auth, input validation, injection, secrets, TLS, SSRF, container hardening
- **Reliability** — retries, circuit breakers, timeouts, health probes, rate limiting, PDB, HPA
- **Code Quality** — type safety, test coverage, logging, documentation, Literal types, OpenAPI accuracy
- **Ops & Deployment** — dependency pinning, image tags, structured logging, metrics, manifests

If an item spans categories, put it in the most impactful one.

### Guidelines

- **Stable IDs.** Technical debt items get `TD-N` IDs. Once assigned, an ID should not change between runs — append new items at the end. This lets the user reference `TD-3` in conversation and have it mean the same thing next week.
- **Omit empty sections.** If no items fall in a category, skip that section.
- **Don't editorialize beyond the source.** The rationale column should reflect what was decided during triage, not new opinions. The source review has the full context.
- **Link to source.** The Source column should name the archive slug (e.g., `2026-04-03_initial-build`) or `staging` for unarchived reviews, so the user can find the original review for details.
