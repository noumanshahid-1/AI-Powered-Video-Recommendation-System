# run.ps1 — open browser when API is live, then run uvicorn (shows logs here)
$PORT = 8000
$BASE = "http://127.0.0.1:$PORT"
$HEALTH = "$BASE/health"

# Start a hidden helper PowerShell that polls /health and opens the browser once
Start-Process powershell -WindowStyle Hidden -NoProfile -ArgumentList @"
  while ($true) {
    try {
      \$r = Invoke-WebRequest -Uri '$HEALTH' -UseBasicParsing -TimeoutSec 1
      if (\$r.StatusCode -eq 200) {
        Start-Sleep -Seconds 1
        Start-Process '$BASE/'
        break
      }
    } catch {}
    Start-Sleep -Milliseconds 500
  }
"@

# Run the API in the current window (you keep the logs)
uvicorn api.main:app --reload --port $PORT
