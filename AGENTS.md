# Rules for AI coding helpers (Codex, Claude)

This is Bidhan's BrAPI breeding data assistant. Bidhan is learning: explain in plain words and keep reports short.

## Branches and releases
- `codex/initial-release` is the published line. Tag `v1.0` marks the exact version running on Hugging Face.
- Work only on `dev`, or on a branch made from `dev`.
- Publishing happens only when Bidhan says so for that release: merging into the published line, adding a tag (v1.1, v1.2, ...), or uploading to the Hugging Face Space `bidhanghimire/brapi-breeding-assistant`.
- Never force-push. Never delete or move branches, tags or repositories.

## Ask Bidhan first
- Commits, pushes, installing packages, live BrAPI requests, real-model (OpenAI) runs, deleting files, changing settings.

## Never
- Read, print or commit `.env` files, API keys or other secrets.

## Every task
- One task at a time. Show the plan and the diff before changing files, run the offline tests, then report what changed, the commands and results, and what is still unverified. Then stop.
