#!/bin/bash

# Advanced Dengue Guard AI Backend Setup Script
echo "🦟 Setting up Advanced Dengue Guard AI Backend..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the root of the dengue-guard-my project"
    exit 1
fi

# Install Python dependencies for the advanced backend
echo "📦 Installing Python dependencies for Advanced AI backend..."
cd backend_advanced

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install Python packages
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment template
if [ ! -f ".env" ]; then
    echo "⚙️ Setting up environment variables..."
    cp .env.template .env
    echo "✏️ Please edit .env file with your configuration"
fi

echo ""
echo "✅ Advanced Dengue Guard AI Backend setup complete!"
echo ""
echo "🚀 To start the Advanced AI server:"
echo "   cd backend_advanced"
echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "📚 API Documentation will be available at:"
echo "   http://localhost:8000/docs (Swagger UI)"
echo "   http://localhost:8000/redoc (ReDoc)"
echo ""
echo "🌐 Frontend will connect to: http://localhost:8000"