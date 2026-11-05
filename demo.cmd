@echo off
REM Windows demo: submit a recording to the async ingest endpoint and print the job info.
REM Usage: demo.cmd C:\path\to\session.mp4
REM No sample video ships with the repo. For the full flow (poll -> summary -> ask) see README, or use demo.sh from WSL/Git Bash.
if "%~1"=="" (
  echo Usage: %~nx0 C:\path\to\session.mp4
  exit /b 2
)
if not exist "%~1" (
  echo File not found: %~1
  exit /b 2
)
if "%BASE_URL%"=="" set BASE_URL=http://localhost:8082
curl.exe -sS -X POST "%BASE_URL%/v2/ingest-video/" -F "file=@%~1"
echo.
echo Poll the returned status_url, e.g.: curl.exe %BASE_URL%/jobs/^<job_id^>
