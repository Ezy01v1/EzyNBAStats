@echo off
rem Daily job: score yesterday's predictions, then record today's (run by Task Scheduler).
rem Output is appended to logs\daily_predictions.log in the repo.
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set PYTHONUTF8=1
set LOG=logs\daily_predictions.log
echo ==== %date% %time% ==== >> "%LOG%"
".venv\Scripts\python.exe" scripts\score_predictions.py >> "%LOG%" 2>&1
".venv\Scripts\python.exe" scripts\record_predictions.py >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo exit %RC% >> "%LOG%"
exit /b %RC%
