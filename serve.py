"""Start the XWIDS analyst dashboard.

    python serve.py                      # http://127.0.0.1:5000  (dataset from XWIDS_DATASET, default synthetic)
Environment: XWIDS_DATASET, XWIDS_USERS="user:pass,...", XWIDS_SECRET_KEY, XWIDS_API_KEY, PORT, HOST, XWIDS_RESET=1
"""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "5000"))
    try:
        from waitress import serve
        print(f"XWIDS dashboard on http://{host}:{port}  (waitress)")
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        print(f"XWIDS dashboard on http://{host}:{port}  (Flask dev server; pip install waitress for production)")
        app.run(host=host, port=port, threaded=True)
