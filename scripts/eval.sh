#!/usr/bin/env bash
# Named recipes for the `eval tool-calling` GitHub Actions workflow.
# Wraps `gh workflow run` so you don't have to remember the axis
# names and valid combinations. Requires `gh` authenticated against
# the rromenskyi/mcp-weather-simple repo.
#
# Usage:
#   scripts/eval.sh <recipe>
#
# Example:
#   scripts/eval.sh schema-ab    # full A/B experiment, 8 parallel rows
#   scripts/eval.sh quick        # single 7b row, fast iteration
#
# Run `scripts/eval.sh help` for the full list.

set -euo pipefail

WORKFLOW="eval tool-calling"

usage() {
    cat <<'EOF'
Recipes (pass one as the first argument):

  quick        One 7b row, full suite (chunk_count=1), no schema.
               Fastest path — ~15 min wall-clock. Use for iterating
               on docstring tweaks when you only need the 7b signal.

  matrix       Default matrix: 7b + 14b, 2 chunks, schema=off.
               Matches nightly scheduled runs. ~25 min wall-clock
               (bounded by slowest row).

  schema-ab    The outputSchema A/B experiment: 7b + 14b × off + on,
               2 chunks = 8 parallel rows. Compare per-family hit
               rate AND latency deltas between schema modes.

  full         Widest fanout: 7b + 14b × off + on × 4 chunks = 16
               parallel rows. Use when 14b is bumping ceilings on
               `schema-ab` (each row then owns ~10 cases).

  14b-only     Single 14b row, 2 chunks, schema=off. Useful when 7b
               is green and you only want the slower row's numbers.

  help         Print this message.

All recipes print the run URL; follow it for step summaries.
EOF
}

case "${1:-help}" in
    quick)
        gh workflow run "$WORKFLOW" \
            -f model=qwen2.5:7b \
            -f chunk_count=1
        ;;
    matrix)
        gh workflow run "$WORKFLOW" \
            -f chunk_count=2
        ;;
    schema-ab)
        gh workflow run "$WORKFLOW" \
            -f output_schema=both \
            -f chunk_count=2
        ;;
    full)
        gh workflow run "$WORKFLOW" \
            -f output_schema=both \
            -f chunk_count=4
        ;;
    14b-only)
        gh workflow run "$WORKFLOW" \
            -f model=qwen2.5:14b \
            -f chunk_count=2
        ;;
    help|-h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown recipe: ${1}" >&2
        echo >&2
        usage >&2
        exit 2
        ;;
esac

# The `run` command returns before the workflow actually starts —
# print a link to the runs list so the user can follow along.
echo
echo "Triggered. Follow progress:"
echo "  gh run watch --exit-status"
echo "  or open: https://github.com/rromenskyi/mcp-weather-simple/actions/workflows/eval.yml"
