@echo off
REM Quick start script for HieraCascade training (Windows)
REM 
REM Usage:
REM   run_hieracascade.bat [fold_number]
REM
REM Example:
REM   run_hieracascade.bat 0

SET FOLD=%1
IF "%FOLD%"=="" SET FOLD=0

SET DATA_ROOT=data
SET SHEET_CSV=data\sheet.csv
SET OUTPUT_DIR=outputs\hieracascade

echo ========================================
echo HieraCascade Training Pipeline
echo ========================================
echo Fold: %FOLD%
echo Data root: %DATA_ROOT%
echo Sheet CSV: %SHEET_CSV%
echo Output dir: %OUTPUT_DIR%
echo.

REM Check if Python is available
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Error: Python not found. Please install Python 3.8+
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
pip install -q -r hieracascade\requirements.txt

REM Run the pipeline
echo.
echo Starting training pipeline...
python -m hieracascade.quick_start ^
    --data_root %DATA_ROOT% ^
    --sheet_csv %SHEET_CSV% ^
    --output_dir %OUTPUT_DIR% ^
    --fold %FOLD% ^
    --device cuda

echo.
echo ========================================
echo Training complete!
echo ========================================
echo Results saved to: %OUTPUT_DIR%
echo.
echo Next steps:
echo 1. View training curves: %OUTPUT_DIR%\stage2\fold%FOLD%\plots\training_curves.png
echo 2. View saliency maps: %OUTPUT_DIR%\stage1\fold%FOLD%\visualizations\
echo 3. Evaluate model:
echo    python -m hieracascade.evaluate ^
echo      --checkpoint %OUTPUT_DIR%\stage2\fold%FOLD%\checkpoint_best.pt ^
echo      --stage stage2 ^
echo      --data_root %DATA_ROOT% ^
echo      --labels_csv %OUTPUT_DIR%\labels.csv ^
echo      --stage1_ckpt %OUTPUT_DIR%\stage1\fold%FOLD%\checkpoint_best.pt ^
echo      --output_dir %OUTPUT_DIR%\stage2\fold%FOLD%\eval ^
echo      --fold %FOLD%

pause
