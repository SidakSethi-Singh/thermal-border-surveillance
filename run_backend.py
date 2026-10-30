import sys
import os
from pathlib import Path

# Auto-activate .venv if running from system python
root_dir = Path(__file__).resolve().parent
venv_python = root_dir / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and sys.executable.lower() != str(venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([str(venv_python)] + sys.argv))

import uvicorn

if __name__ == "__main__":
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    print("==================================================================")
    print("  AEGIS-FLIR Surveillance System - FastAPI Backend Starting")
    print("  Web Dashboard: http://localhost:8000")
    print("  API Docs:      http://localhost:8000/docs")
    print("==================================================================")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
