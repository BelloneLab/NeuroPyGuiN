#!/usr/bin/env bash
# Install a private Python runtime and a normal Git checkout, without Conda setup.
set -euo pipefail

usage() {
    cat <<'HELP'
NeuroPyGuiN installer (Linux x86_64)
  bash Install-NeuroPyGuiN.sh [--prefix DIRECTORY] [--source CHECKOUT]
                            [--update] [--no-launch]
The default location is ~/.local/share/NeuroPyGuiN (or $XDG_DATA_HOME).
Run this file inside a checkout to use that checkout. Otherwise main is cloned.
Re-running repairs dependencies and recreates shortcuts. --update first pulls
origin/main, fast-forward only, and refuses a checkout with local changes.
HELP
}

prefix="${XDG_DATA_HOME:-$HOME/.local/share}/NeuroPyGuiN"
source_dir=""
update=0
launch=1
terminal=0
original_args=("$@")
while (($#)); do
    case "$1" in
        --prefix|--source)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            if [[ "$1" == --prefix ]]; then prefix="$2"; else source_dir="$2"; fi
            shift 2 ;;
        --update) update=1; shift ;;
        --no-launch) launch=0; shift ;;
        --terminal) terminal=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    echo 'This installer supports Linux x86_64. Use the Windows installer on Windows.' >&2; exit 1;
}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# File-manager launches need a visible terminal for download progress and errors.
if [[ ! -t 1 && $terminal == 0 && -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    for term in x-terminal-emulator gnome-terminal konsole xfce4-terminal xterm; do
        if command -v "$term" >/dev/null; then
            case "$term" in
                gnome-terminal) exec "$term" -- bash "$script_dir/$(basename -- "$0")" --terminal "${original_args[@]}" ;;
                xfce4-terminal) exec "$term" --execute bash "$script_dir/$(basename -- "$0")" --terminal "${original_args[@]}" ;;
                *) exec "$term" -e bash "$script_dir/$(basename -- "$0")" --terminal "${original_args[@]}" ;;
            esac
        fi
    done
fi
finish() {
    code=$?
    if ((code)); then echo "Installation stopped (exit $code). Re-run this installer to retry." >&2; fi
    if ((terminal)) && [[ -t 0 ]]; then read -r -p 'Press Enter to close...' _ || true; fi
    exit "$code"
}
trap finish EXIT
mkdir -p -- "$prefix"
prefix="$(cd -- "$prefix" && pwd)"
# Both the binary version and checksum are pinned to the official release.
mamba="$prefix/bin/micromamba"
expected=fb4554d61a1c567726890169e39c41aa8495dbba56f50957ee7356572b6c5726
mkdir -p -- "$prefix/bin"
if [[ ! -f "$mamba" ]] || [[ "$(sha256sum "$mamba" | cut -d ' ' -f 1)" != "$expected" ]]; then
    echo 'Downloading the private environment manager...'
    url=https://github.com/mamba-org/micromamba-releases/releases/download/2.7.0-0/micromamba-linux-64
    if command -v curl >/dev/null; then
        curl --fail --location --retry 3 "$url" -o "$mamba.download"
    elif command -v wget >/dev/null; then
        wget -O "$mamba.download" "$url"
    else
        echo 'Install curl or wget, then re-run this installer.' >&2; exit 1
    fi
    [[ "$(sha256sum "$mamba.download" | cut -d ' ' -f 1)" == "$expected" ]] || {
        rm -f -- "$mamba.download"; echo 'Download checksum mismatch.' >&2; exit 1;
    }
    mv -- "$mamba.download" "$mamba"
fi
chmod +x -- "$mamba"
export MAMBA_ROOT_PREFIX="$prefix/mamba"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
if [[ ! -x "$prefix/bootstrap/bin/python" || ! -x "$prefix/bootstrap/bin/git" ]]; then
    "$mamba" create -y --no-rc -p "$prefix/bootstrap" --override-channels -c conda-forge python=3.10 git
fi
export PATH="$prefix/bootstrap/bin:$PATH"
if [[ -z "$source_dir" ]]; then
    if [[ -f "$script_dir/main.py" && -f "$script_dir/environment-linux.yml" ]]; then
        source_dir="$script_dir"
    else
        source_dir="$prefix/app"
    fi
fi
if [[ ! -e "$source_dir" ]]; then
    git clone --branch main https://github.com/BelloneLab/NeuroPyGuiN.git "$source_dir"
fi
source_dir="$(cd -- "$source_dir" && pwd)"
[[ -f "$source_dir/install/setup.py" ]] || {
    echo "Not a NeuroPyGuiN checkout: $source_dir" >&2; exit 1;
}
args=(--prefix "$prefix" --source "$source_dir")
if ((update)); then args+=(--update); fi
if ((!launch)); then args+=(--no-launch); fi
"$prefix/bootstrap/bin/python" "$source_dir/install/setup.py" "${args[@]}"
