#!/bin/bash

# Immediately exit if any command has a non-zero exit status
set -e

APPLY_PATCH_OPTION="-a"
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --no-apply-patches)
      APPLY_PATCH_OPTION=""
      shift
      ;;
    --diff-only)
      CHECK_DIFF_ONLY="1"
      shift
      ;;
    *) echo "[ERROR] Unknown parameter passed: $1"; exit 255 ;;
  esac
done

GIT_ROOT="$(git rev-parse --show-toplevel)"
STYLE_DIR="${GIT_ROOT}/infra/style"

check_ruff_is_installed() {
  if ! command -v ruff >/dev/null 2>&1; then
    echo "Error: ruff command not found." >&2
    exit 1
  fi
}

check_ruff_fixes() {
  output="$(ruff check --fix 2>&1 || true)"

  if $(echo $output | tail -n 1 | grep -q "No fixes available"); then
    echo "🟢 Ruff가 수정한 항목이 없습니다."
    return 0
  else
    echo "✅ Ruff가 파일을 수정했습니다."
    exit 1
  fi
}


check_ruff_is_installed
check_ruff_fixes


if [[ "${CHECK_DIFF_ONLY}" = "1" ]]; then
  MAIN_EXIST=$(git rev-parse --verify main)
  CURRENT_BRANCH=$(git branch | grep \* | cut -d ' ' -f2-)
  DIFF_COMMITS=`git log --graph --oneline main..HEAD | wc -l`
  if [[ -z "${MAIN_EXIST}" ]]; then
    echo "Cannot find main branch"
    exit 1
  elif [[ "${CURRENT_BRANCH}" = "main" ]]; then
    echo "Current branch is main"
    exit 1
  else
    # Gather diff from HEAD
    FILES_TO_CHECK=$(git diff --name-only --diff-filter=d HEAD~${DIFF_COMMITS})

    # Remove links
    # Git file mode
    #   120000: symbolic link
    #   160000: git link
    # Reference: https://github.com/git/git/blob/cd42415/Documentation/technical/index-format.txt#L72-L81
    FILES_TO_CHECK=$(git ls-files -c -s --exclude-standard ${FILES_TO_CHECK[@]} | egrep -v '^1[26]0000' | cut -f2)
  fi
  lintrunner --force-color $APPLY_PATCH_OPTION --config "${GIT_ROOT}/.lintrunner.toml" $FILES_TO_CHECK
else
  lintrunner --force-color --all-files $APPLY_PATCH_OPTION --config "${GIT_ROOT}/.lintrunner.toml"
fi
