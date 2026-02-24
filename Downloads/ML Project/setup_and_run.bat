@echo off
echo ============================================
echo  Fake Product Hype Detection - Setup
echo ============================================

echo.
echo [1/4] Creating virtual environment...
python -m venv hype_env
if errorlevel 1 (echo ERROR: Python not found. Install Python 3.10+ first. && pause && exit /b 1)

echo.
echo [2/4] Activating environment...
call hype_env\Scripts\activate

echo.
echo [3/4] Installing packages...
python -m pip install --upgrade pip -q
pip install -r requirements.txt
if errorlevel 1 (echo ERROR: pip install failed. && pause && exit /b 1)

echo.
echo [4/4] Generating sample data...
python run_pipeline.py
if errorlevel 1 (echo ERROR: Pipeline failed. Check errors above. && pause && exit /b 1)

echo.
echo ============================================
echo  Setup COMPLETE!
echo  Run:  streamlit run app.py
echo ============================================
pause
