>> Run the following commands in the fresh new terminal one by one.
1. python3 -m venv .venv
2. source .venv/bin/activate
3. pip install -r api/requirements.txt

# (optional) set artifacts path; default is ./artifacts
export RECSYS_ARTIFACTS="$(pwd)/artifacts"

>> For refreshing the application launcher
4. uvicorn api.main:app --reload --port 8000

>> optional-- incase the above the 1 - 3 lines of codes didn't work.
<!-- 5. py -m venv .venv
6. .\.venv\Scripts\Activate.ps1
7. pip install -r requirements.txt -->

>> To Launch the web interface, execute this single command in the same terminal
8. uvicorn api.main:app --reload --port 8000

# or
.\.venv311\Scripts\activate
uvicorn api.main:app --reload --port 8000
