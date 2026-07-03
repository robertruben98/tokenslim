#!/bin/sh
# install.sh — install the TokenSlim CLI and library.
#
# Usage:
#   sh install.sh                # install tokenslim
#   sh install.sh --with-extras  # install tokenslim[tokenizers,images,semantic]
#
# Behavior:
#   * Uses pipx when available (isolated CLI install), otherwise
#     `pip install --user`.
#   * Idempotent: re-running upgrades/reinstalls the same package.
#   * Verifies the install by running `tokenslim doctor`.
#
# This script performs no downloads of its own beyond the package manager
# invocation — no curl-pipe-to-shell.

set -eu

SPEC="tokenslim"
EXTRAS_SPEC="tokenslim[tokenizers,images,semantic]"

usage() {
    printf '%s\n' \
        "Usage: sh install.sh [--with-extras]" \
        "" \
        "  --with-extras  Install optional extras too:" \
        "                 ${EXTRAS_SPEC}" \
        "  -h, --help     Show this help."
}

for arg in "$@"; do
    case "$arg" in
        --with-extras)
            SPEC="$EXTRAS_SPEC"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'install.sh: unknown option: %s\n' "$arg" >&2
            usage >&2
            exit 1
            ;;
    esac
done

find_python() {
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

echo "Installing ${SPEC} ..."

if command -v pipx >/dev/null 2>&1; then
    # --force makes re-runs idempotent (reinstall/upgrade instead of erroring).
    pipx install --force "$SPEC"
    INSTALLER="pipx"
else
    PY="$(find_python)" || {
        echo "install.sh: no python3/python found on PATH. Install Python 3.10+ first." >&2
        exit 1
    }
    "$PY" -m pip install --user --upgrade "$SPEC"
    INSTALLER="pip --user"
fi

echo "Installed via ${INSTALLER}. Verifying with 'tokenslim doctor' ..."

if command -v tokenslim >/dev/null 2>&1; then
    tokenslim doctor
elif PY="$(find_python)" && "$PY" -m tokenslim.cli doctor; then
    echo "Note: 'tokenslim' is not on PATH yet (check ~/.local/bin)." >&2
else
    echo "install.sh: verification failed — 'tokenslim doctor' did not run." >&2
    echo "If you installed with pip --user, add ~/.local/bin to PATH and retry." >&2
    exit 1
fi

echo "TokenSlim is ready. Try: tokenslim --help"
