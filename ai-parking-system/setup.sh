#!/bin/bash

# AI Parking Management System Setup Script
# This script sets up the complete development environment

set -e

echo "🚗 Setting up AI Parking Management System..."
echo "=============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p uploads
mkdir -p logs
mkdir -p data/postgres
mkdir -p data/redis

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set up frontend
echo "⚛️  Setting up React frontend..."
cd frontend
npm install
cd ..

# Set up environment variables
echo "🔧 Setting up environment variables..."
if [ ! -f .env ]; then
    cat > .env << 'ENVEOF'
# AI Parking Management System Environment Variables

# Application
APP_NAME=AI Parking Management System
DEBUG=true
SECRET_KEY=your-secret-key-here-change-in-production

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/parking_db
REDIS_URL=redis://localhost:6379

# ML Models
YOLO_MODEL_PATH=ml_models/yolov5_parking.pt
KNN_MODEL_PATH=ml_models/knn_classifier.pkl

# External APIs
PAYMENT_API_KEY=your-payment-api-key
NAVIGATION_API_KEY=your-navigation-api-key

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
ENVEOF
    echo "✅ Created .env file"
else
    echo "✅ .env file already exists"
fi

# Build and start services
echo "�� Building and starting Docker services..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "�� Setup completed successfully!"
echo ""
echo "📱 Access your applications:"
echo "   • Web App: http://localhost:3000"
echo "   • API Docs: http://localhost:8000/docs"
echo "   • Monitoring: http://localhost:3001"
echo ""
echo "📚 Next steps:"
echo "   1. Open http://localhost:3000 in your browser"
echo "   2. Check the API documentation at http://localhost:8000/docs"
echo "   3. Monitor system metrics at http://localhost:3001"
echo ""
echo "🛠️  Development commands:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop services: docker-compose down"
echo "   • Restart services: docker-compose restart"
echo "   • Update code: docker-compose up -d --build"
echo ""
echo "Happy coding! 🚗💻"
