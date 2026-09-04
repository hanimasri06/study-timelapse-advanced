import hashlib
import os
import subprocess
import sys


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(root, "requirements.txt")
    marker_path = os.path.join(root, ".venv", ".requirements.sha256")

    with open(requirements_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    installed_digest = ""
    try:
        with open(marker_path, "r", encoding="ascii") as f:
            installed_digest = f.read().strip()
    except OSError:
        pass

    if installed_digest == digest:
        print("[INFO] Dependencies are up to date.")
        return

    print("[INFO] Installing updated dependencies...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "-r",
        requirements_path,
        "--quiet",
    ])
    with open(marker_path, "w", encoding="ascii") as f:
        f.write(digest)


if __name__ == "__main__":
    main()
