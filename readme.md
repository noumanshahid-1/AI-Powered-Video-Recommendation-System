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

## UI Preview
### 🔹 Homepage
<img width="1920" height="1080" alt="Homepage" src="https://github.com/user-attachments/assets/941f7fe9-34f2-4814-907d-32e94a6dcd8f" />
### 🔹 Recommendations
<img width="1920" height="1080" alt="result_page" src="https://github.com/user-attachments/assets/6960d3a1-5677-443d-a088-67838245999a" />

