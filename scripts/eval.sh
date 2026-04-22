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

  quick        One 9b row, 4 chunks, no schema. ~6 parallel rows
               after shard fanout. Fastest small-model signal. ~15 min
               wall-clock.

  matrix       Default: 9b + 14b × 4 chunks, schema=off. Nightly
               shape. 8 parallel rows, ~20-25 min wall-clock.

  schema-ab    outputSchema A/B: 9b + 14b × off + on × 4 chunks =
               16 parallel rows. Under the 20-concurrent-job cap.
               Compares per-family hit rate AND latency deltas.

  docstring-ab Docstring verbosity A/B: 9b + 14b × terse + verbose
               × 4 chunks = 16 rows. Same shape as schema-ab.

  full         Widest safe: 9b + 14b × schema=both × chunks=4 =
               16 rows. Adding docstrings=both pushes to 32 rows
               which queues — use `full-wide` if you need it.

  full-wide    Every axis both, 8 chunks: 2 × 2 × 2 × 8 = 64 rows.
               Will partially queue on the 20-job cap, but each
               row runs fast and the whole thing completes in
               ~2-3x the slowest row's wall-clock. Use rarely.

  14b-only     Single 14b row, 4 chunks, schema=off. Useful when
               9b is green and only the slower row needs numbers.

  help         Print this message.

All recipes print the run URL; follow it for step summaries.
EOF
}

case "${1:-help}" in
    quick)
        gh workflow run "$WORKFLOW" \
            -f model=qwen3.5:9b \
            -f chunk_count=4
        ;;
    matrix)
        gh workflow run "$WORKFLOW" \
            -f chunk_count=4
        ;;
    schema-ab)
        gh workflow run "$WORKFLOW" \
            -f output_schema=both \
            -f chunk_count=4
        ;;
    docstring-ab)
        gh workflow run "$WORKFLOW" \
            -f docstrings=both \
            -f chunk_count=4
        ;;
    full)
        gh workflow run "$WORKFLOW" \
            -f output_schema=both \
            -f chunk_count=4
        ;;
    full-wide)
        gh workflow run "$WORKFLOW" \
            -f output_schema=both \
            -f docstrings=both \
            -f chunk_count=8
        ;;
    14b-only)
        gh workflow run "$WORKFLOW" \
            -f model=qwen3:14b \
            -f chunk_count=4
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
