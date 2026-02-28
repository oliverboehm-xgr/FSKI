@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM Bunny / FSKI runtime config (reproducible)
REM ==========================================================

REM --- MUST: Full pipeline (Organe laufen) ---
set BUNNY_LITE=0

REM --- Context window (global) ---
REM Soft-mode: keep CTX moderate to avoid crashes on CPU/RAM limited machines.
set BUNNY_CTX=8192

REM --- Organ activation thresholds/cooldowns ---
REM These thresholds are now *real* cutoffs: above threshold, an organ runs deterministically.
REM Keep them moderate, otherwise daydream/evolve will never run and logs stay empty.
set BUNNY_TH_WEBSENSE=0.60
set BUNNY_TH_DAYDREAM=0.60
set BUNNY_TH_EVOLVE=0.70
set BUNNY_TH_AUTOTALK=0.85
set BUNNY_AUTOTALK_MAX_ERROR_SIGNAL=0.15
set BUNNY_AUTOTALK_MAX_WS_NEED=0.25
set BUNNY_AUTOTALK_COOLDOWN=600
set BUNNY_IDLE_PERIOD=30
set BUNNY_IDLE_COOLDOWN=180
set BUNNY_EVOLVE_COOLDOWN=3600
set BUNNY_PROPOSAL_REFINE_COOLDOWN=1800
set BUNNY_WORKSPACE_MAX=10

REM --- Output length control (token budget) ---
REM Prevent rambling / repetition by limiting generation length.
set BUNNY_NUM_PREDICT=240

REM --- Epistemic sensor blending (evidence -> uncertainty/confidence) ---
set BUNNY_ETA_EPISTEMIC=0.45

REM --- Memory retention / soft forgetting ---
set BUNNY_MEMORY_SHORT_MAX=800
set BUNNY_MEMORY_SHORT_TTL_DAYS=30
set BUNNY_BELIEF_HALF_LIFE_DAYS=45
REM Salience -> stickiness gain (emotion/impact). Higher = slower forgetting for high-salience beliefs.
set BUNNY_SALIENCE_HALF_LIFE_GAIN=1.2


REM --- Episode tagging (tag-and-capture memory around high-salience events) ---
set BUNNY_EPISODE_TH=0.72
set BUNNY_EPISODE_WINDOW=6
set BUNNY_EPISODE_TAU=2.5
set BUNNY_BELIEF_FLOOR=0.15
set BUNNY_BELIEF_PRUNE_TTL_DAYS=180
set BUNNY_BELIEF_PRUNE_BELOW=0.18

REM --- Policy kernel (trainable action prior; daydream can mutate)
set BUNNY_POLICY_ENABLE=1
set BUNNY_POLICY_ETA=0.05
set BUNNY_POLICY_L2=0.001
set BUNNY_POLICY_MAXABS=3.0
set BUNNY_POLICY_FROB=25.0

REM --- WebSense -> Belief assimilation (autonomous learning)
REM Set to 0 to disable if memory becomes noisy.
set BUNNY_WEBSENSE_ASSIMILATE=1
set BUNNY_WEBSENSE_ASSIMILATE_MAX=4

REM --- Belief extractor tuning ---
set BUNNY_CTX_BELIEFS=2048
set BUNNY_TEMP_BELIEFS=0.2

REM --- Models (adjust to your installed Ollama models) ---
REM Note: --model CLI below overrides BUNNY_MODEL if your code prioritizes args.
set BUNNY_MODEL=llama3.2:3b
set BUNNY_MODEL_SPEECH=llama3.1:8b
set BUNNY_MODEL_DECIDER=llama3.2:3b
set BUNNY_MODEL_FEEDBACK=llama3.2:3b
set BUNNY_MODEL_EVIDENCE=llama3.2:3b
set BUNNY_MODEL_TOPIC=llama3.2:3b
set BUNNY_MODEL_WEBSENSE=llama3.2:3b
set BUNNY_MODEL_BELIEFS=llama3.2:3b
set BUNNY_MODEL_SELFEVAL=llama3.2:3b

REM ==========================================================
REM Diagnostics (so you see what actually applied)
REM ==========================================================
echo.
echo [BUNNY] LITE=%BUNNY_LITE%  CTX=%BUNNY_CTX%
echo [BUNNY] MODEL=%BUNNY_MODEL%
echo [BUNNY] SPEECH=%BUNNY_MODEL_SPEECH%
echo [BUNNY] DECIDER=%BUNNY_MODEL_DECIDER%
echo [BUNNY] FEEDBACK=%BUNNY_MODEL_FEEDBACK%
echo [BUNNY] EVIDENCE=%BUNNY_MODEL_EVIDENCE%
echo [BUNNY] TOPIC=%BUNNY_MODEL_TOPIC%
echo [BUNNY] WEBSENSE=%BUNNY_MODEL_WEBSENSE%
echo [BUNNY] BELIEFS=%BUNNY_MODEL_BELIEFS%
echo [BUNNY] TH_WEBSENSE=%BUNNY_TH_WEBSENSE% TH_DAYDREAM=%BUNNY_TH_DAYDREAM% TH_EVOLVE=%BUNNY_TH_EVOLVE%
echo [BUNNY] IDLE_PERIOD=%BUNNY_IDLE_PERIOD% IDLE_COOLDOWN=%BUNNY_IDLE_COOLDOWN%
echo.

REM ==========================================================
REM Start
REM ==========================================================
REM If you use a venv, activate it here:
REM call .venv\Scripts\activate.bat

REM NOTE: run via Python sources (start.cmd). main.exe may be an older frozen build.
REM Use the bundled DB by default (keeps learning across restarts).
python -m app.ui --db data\frankenstein.sqlite --addr 127.0.0.1:8080

echo.
echo [BUNNY] process exited with code %ERRORLEVEL%
pause
endlocal