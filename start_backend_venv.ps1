# Start Backend Server with Virtual Environment
# This script ensures the virtual environment is activated and server runs from correct directory

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Starting Backend with Virtual Environment..." -ForegroundColor Green
Write-Host "URL: http://localhost:8001" -ForegroundColor Yellow
Write-Host "Docs: http://localhost:8001/docs" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

# Change to script directory
Set-Location -Path $PSScriptRoot

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Start uvicorn
uvicorn backend.api_gateway:app --reload --port 8001

