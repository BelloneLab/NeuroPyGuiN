from __future__ import annotations

import sys
import traceback


def _entry() -> int:
    try:
        from neuropyguin.app import main
    except Exception as exc:
        msg = str(exc)
        if "PySide6" in msg or "QtCore" in msg or "DLL load failed" in msg:
            print("Failed to import PySide6/Qt runtime.", file=sys.stderr)
            print(f"Python executable: {sys.executable}", file=sys.stderr)
            print("This usually means you are launching with a different Python than the one where PySide6 is installed.", file=sys.stderr)
            print("Use the intended conda env, for example:", file=sys.stderr)
            print("  conda activate ks4_ece", file=sys.stderr)
            print("  python main.py", file=sys.stderr)
            return 2
        traceback.print_exc()
        return 1
    return int(main())


if __name__ == "__main__":
    raise SystemExit(_entry())
