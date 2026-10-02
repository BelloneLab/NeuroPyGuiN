# Install and update NeuroPyGuiN

## Quick start

- **Windows x64:** download the root `Install-NeuroPyGuiN.bat`, then double-click it.
  Keep its console open during installation. No preinstalled Python, Git, or Conda is required.
- **Linux x86_64:** save `Install-NeuroPyGuiN.desktop` on your desktop, mark it
  executable or choose **Allow Launching**, and open it. This downloads and runs
  the shell installer. Alternatively, download `Install-NeuroPyGuiN.sh`, then run
  `bash Install-NeuroPyGuiN.sh`. It uses curl or wget and standard Linux utilities.
  When launched through a file manager, it opens a terminal for progress and errors.

Running the installer from a source checkout uses that checkout. Running a standalone
download clones `https://github.com/BelloneLab/NeuroPyGuiN.git`, branch `main`.
The installer never adds itself to your shell profile or changes existing Conda environments.

The scientific stack includes CUDA-enabled PyTorch and takes several GB to download.
GPU sorting requires a compatible NVIDIA driver. Atlas downloads and optional C4
classification are configured separately, as described in the main README.

## What gets installed

| Item | Linux | Windows |
| --- | --- | --- |
| Default installation folder | `$XDG_DATA_HOME/NeuroPyGuiN`, or `~/.local/share/NeuroPyGuiN` | `%LOCALAPPDATA%\NeuroPyGuiN` |
| Source (standalone installer) | `app/` | `app\` |
| Isolated app environment | `runtime/` | `runtime\` |
| Python and Git for installation | `bootstrap/` | `bootstrap\` |
| Environment manager and cache | `bin/`, `mamba/` | `bin\`, `mamba\` |
| Installation paths | `installation.json` | `installation.json` |
| Shortcut startup log | `logs/launcher.log` | `logs\launcher.log` |

The environment manager is [micromamba](https://mamba.readthedocs.io/en/stable/installation/micromamba-installation.html),
pinned to release `2.7.0-0`. Each platform's binary is checked against its SHA-256
before it is executed. App dependencies come from the same platform environment
YAML used for manual Conda installation.

**NeuroPyGuiN** and **Update NeuroPyGuiN** appear in the applications or Start menu.
Desktop copies are created when a desktop directory exists. Linux desktops may
require right-clicking a new shortcut and choosing **Allow Launching**. Headless
Linux installations receive menu files without creating a desktop directory.
Windows shortcuts use a multi-resolution `.ico` rendered from the SVG master.

The launch shortcut activates the managed runtime, sets the checkout as the working
directory, and records console output in `logs/launcher.log`. It requires no internet
connection after installation. Moving the source or runtime breaks stored paths;
re-run the installer with the new locations to recreate shortcuts.

## Custom locations and unattended setup

Linux:

```bash
bash Install-NeuroPyGuiN.sh --prefix "$HOME/Applications/NeuroPyGuiN" --no-launch
bash Install-NeuroPyGuiN.sh --prefix "$HOME/Applications/NeuroPyGuiN" --source "$HOME/code/NeuroPyGuiN"
```

Windows PowerShell, from a clone:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install/install.ps1 -Prefix "D:\Apps\NeuroPyGuiN" -NoLaunch
powershell -NoProfile -ExecutionPolicy Bypass -File install/install.ps1 -Prefix "D:\Apps\NeuroPyGuiN" -Source "D:\Code\NeuroPyGuiN"
```

The Windows `.bat` also forwards `-Prefix`, `-Source`, `-Update`, and `-NoLaunch`.
It pauses at the end so double-click users can read the result. Call PowerShell
directly for unattended setup. Use a separate prefix for each independent installation.
Do not run two installers against the same prefix at once.

## Update and repair

Close the app and use the **Update NeuroPyGuiN** shortcut, or run:

```bash
bash Install-NeuroPyGuiN.sh --update --no-launch
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install/install.ps1 -Update -NoLaunch
```

Supply the same `--prefix` / `-Prefix` and `--source` / `-Source` for a custom
installation. Generated update shortcuts remember both automatically.

Updates require `main`, a clean tracked working tree, and a fast-forward from
`origin/main`. Local commits and edits are never reset or stashed. Git protects
untracked files that conflict with incoming changes. Resolve any reported conflict
in your checkout, then retry. ZIP source downloads can be installed, but Git updates
require a real clone.

Re-running without the update option repairs dependencies and shortcuts using the
current source revision. If dependency installation fails after a Git update, the
checkout has advanced; fix the reported error and rerun to finish the repair.

Manual Git and Conda installation continues to work exactly as before. For a manual
Conda environment, use `git pull --ff-only origin main`, then the matching
`conda env update -f environment-<platform>.yml --prune` command in the main README.
Avoid updating an environment while the app or a sorting job is running.

## Troubleshooting

- **Download interrupted:** rerun the installer. It reuses installed packages and
  checks the environment manager download before executing it.
- **Launch fails:** read `logs/launcher.log` in the installation folder. Startup
  failures also show a dialog when a system dialog utility is available.
- **Linux Qt/xcb libraries missing:** Debian/Ubuntu systems may need:

  ```bash
  sudo apt install libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \
    libxcb-shape0 libxcb-xinerama0 libxkbcommon-x11-0 libegl1 libopengl0
  ```

  The installer does not change system packages or ask for root. Other Linux
  distributions use their equivalent packages. A graphical desktop is required to
  launch the GUI; `--no-launch` supports preparing a headless installation.
- **Windows CatGT tools:** their native executables may need Microsoft's Visual C++
  Redistributable. Use **Help > Run Diagnostics** for the app's dependency report.
- **External preprocessing tools:** the existing first-launch setup detects missing
  CatGT, TPrime, and C_Waves. They can also be installed from the preprocessing settings.

To uninstall, remove the two menu/desktop shortcuts and the installation folder.
If you used an existing checkout, it is outside the installation folder and remains
untouched. Back up any recordings or settings you deliberately placed inside the
installation folder before deleting it.

## Optional CI checks

`installer-checks.workflow.yml` is a GitHub Actions template for Linux and Windows.
To enable it, copy it to `.github/workflows/installer.yml` using a GitHub credential
with workflow permission. It checks installer behavior, SVG icon export, and script
syntax without downloading the full scientific environment.
