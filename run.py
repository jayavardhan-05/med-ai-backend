import os
import uvicorn

if __name__ == "__main__":
    app_dir = os.path.dirname(os.path.abspath(__file__))
    print("[INFO] Starting MED AI Backend on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, app_dir=app_dir)
