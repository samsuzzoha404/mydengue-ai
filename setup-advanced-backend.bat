@echo off
REM Advanced Dengue Guard AI Backend Setup Script for Windows
echo 🦟 Setting up Advanced Dengue Guard AI Backend...

REM Check if we're in the right directory
if not exist "package.json" (
    echo ❌ Error: Please run this script from the root of the dengue-guard-my project
    pause
    exit /b 1
)

REM Install Python dependencies for the advanced backend
echo 📦 Installing Python dependencies for Advanced AI backend...
cd backend_advanced

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 🐍 Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate

REM Install Python packages
echo 📦 Installing Python packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Copy environment template
if not exist ".env" (
    echo ⚙️ Setting up environment variables...
    copy .env.template .env
    echo ✏️ Please edit .env file with your configuration
)

echo.
echo ✅ Advanced Dengue Guard AI Backend setup complete!
echo.
echo 🚀 To start the Advanced AI server:
echo    cd backend_advanced
echo    venv\Scripts\activate
echo    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 📚 API Documentation will be available at:
echo    http://localhost:8000/docs (Swagger UI)
echo    http://localhost:8000/redoc (ReDoc)
echo.
echo 🌐 Frontend will connect to: http://localhost:8000
pause