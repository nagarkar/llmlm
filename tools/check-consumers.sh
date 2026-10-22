#!/usr/bin/env bash
# The consumer-test gate: run every migrated consumer's test suite against this llmlm working
# tree (consumers.toml). Local mode uses the sibling checkouts and their own .venv; `--ci`
# clones each consumer at main into a temp folder with CONSUMERS_TOKEN and builds a venv.
# Exit 1 when any consumer fails. Scripted tier only: no model is called, nothing bills.
#
#   tools/check-consumers.sh            # local, all migrated consumers
#   tools/check-consumers.sh ytlm beelm # local, only these
#   tools/check-consumers.sh --ci       # CI
#
# A consumer's sibling checkout may be on another branch or behind main. To run its suite from
# another checkout (a worktree holding the change under test), name it:
#   LLMLM_CONSUMER_ROOTS="ytlm=/path/to/a beelm=/path/to/b" tools/check-consumers.sh
set -u
LLMLM="$(cd "$(dirname "$0")/.." && pwd)"
REPOS="$(dirname "$LLMLM")"
MODE=local
ONLY=()
for arg in "$@"; do
  case "$arg" in
    --ci) MODE=ci ;;
    *) ONLY+=("$arg") ;;
  esac
done

rows() {
  python3 - "$LLMLM/consumers.toml" <<'EOF'
import sys, tomllib
data = tomllib.load(open(sys.argv[1], "rb"))
for name, row in data["consumers"].items():
    print(name, row["path"], row.get("subdir", "."), row["repo"], str(row.get("migrated", False)).lower(), row["test"], sep="\t")
EOF
}

status=0
ran=0
while IFS=$'\t' read -r name path subdir repo migrated test; do
  if [ ${#ONLY[@]} -gt 0 ]; then
    case " ${ONLY[*]} " in *" $name "*) ;; *) continue ;; esac
  elif [ "$migrated" != "true" ]; then
    echo "skip   $name: not migrated to llmlm yet (consumers.toml)"
    continue
  fi
  if [ "$MODE" = ci ]; then
    work="$(mktemp -d)"
    if ! git clone --quiet --depth 1 "https://x-access-token:${CONSUMERS_TOKEN:?CONSUMERS_TOKEN is not set}@github.com/$repo.git" "$work/$path"; then
      echo "FAIL   $name: clone of $repo failed"; status=1; continue
    fi
    root="$work/$path/$subdir"
    python3 -m venv "$root/.venv" && "$root/.venv/bin/pip" install -q -e "$LLMLM" -e "$root[dev]" || { echo "FAIL   $name: install failed"; status=1; continue; }
  else
    root="$REPOS/$path/$subdir"
    for override in ${LLMLM_CONSUMER_ROOTS:-}; do
      [ "${override%%=*}" = "$name" ] && root="${override#*=}/$subdir"
    done
    if [ ! -d "$root/.venv" ]; then echo "skip   $name: no checkout or venv at $root"; continue; fi
    "$root/.venv/bin/pip" install -q -e "$LLMLM" || { echo "FAIL   $name: pip install -e llmlm failed"; status=1; continue; }
  fi
  ran=$((ran + 1))
  if (cd "$root" && eval "$test" >"$root/.llmlm-consumer-check.log" 2>&1); then
    echo "ok     $name"
  else
    echo "FAIL   $name: see $root/.llmlm-consumer-check.log"
    tail -20 "$root/.llmlm-consumer-check.log"
    status=1
  fi
done < <(rows)
echo "consumer gate: $ran run, exit $status"
exit $status
