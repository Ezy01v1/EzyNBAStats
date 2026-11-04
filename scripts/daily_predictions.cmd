@echo off
rem Daily job: score earlier predictions, then record today's (run by Task Scheduler).
rem Both steps always run; the job fails if either fails. Output: logs\daily_predictions.log,
rem one-line result of the last run: logs\last_status.txt.
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set PYTHONUTF8=1
set LOG=logs\daily_predictions.log
echo ==== %date% %time% ==== >> "%LOG%"
".venv\Scripts\python.exe" scripts\score_predictions.py >> "%LOG%" 2>&1
set SCORE_RC=%ERRORLEVEL%
".venv\Scripts\python.exe" scripts\record_predictions.py >> "%LOG%" 2>&1
set RECORD_RC=%ERRORLEVEL%
set RC=0
if not "%SCORE_RC%"=="0" set RC=1
if not "%RECORD_RC%"=="0" set RC=1
if "%RC%"=="0" (set STATUS=OK) else (set STATUS=FAILED)
echo STATUS %STATUS% (score=%SCORE_RC% record=%RECORD_RC%) >> "%LOG%"
echo %date% %time% %STATUS% score=%SCORE_RC% record=%RECORD_RC% > logs\last_status.txt
exit /b %RC%
