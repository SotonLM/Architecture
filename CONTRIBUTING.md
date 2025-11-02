# Contributing

Thanks for your interest in improving this project. This guide explains how to get started, how to work locally, and what we expect from pull requests.

## Getting Started

1. Fork the repository and clone your fork locally.
2. Create a new branch from `main` that clearly describes your change, for example `feature/add-auth` or `chore/update-ci`.
3. Install the tooling required for your changes. We standardise on [uv](https://github.com/astral-sh/uv) for dependency management—`uv tool install` and `uv sync` should cover most workflows. If you add new dependencies, document them in the README or relevant configuration file.

## Development Workflow

- Keep commits focused. Small, reviewable changes make it easier to understand your intent.
- Run linting and formatting checks before committing. Keep styles consistent with the existing code (`uv tool run --from ruff ruff .`).
- Run the test suite locally via `uv run pytest` (or `uv tool run --from pytest pytest` if no project environment yet). Add test coverage for new functionality or regression fixes.
- Update documentation, comments, or examples when functionality changes.

## Pull Requests

- Fill out the pull request template completely. Provide context, testing evidence, and any follow-up tasks.
- Keep PRs scoped to a single feature or fix. Large, multi-purpose PRs are difficult to review and maintain.
- Ensure CI checks pass before requesting review.
- Request a review from the appropriate code owners (`.github/CODEOWNERS`) when the PR is ready.

## Commit Message Guidelines

- Use the imperative mood (e.g., "Add linting script" instead of "Added" or "Adds").
- Reference GitHub issues in the body when applicable (e.g., `Fixes #123`).
- Avoid committing generated files or build artefacts unless they are required.

## Reporting Issues

If you spot a bug, please include reproduction steps, expected vs. actual behaviour, and any logs or screenshots that help explain the issue.

## Questions

If you are unsure about anything, open a draft pull request or start a discussion in GitHub to get feedback early.
