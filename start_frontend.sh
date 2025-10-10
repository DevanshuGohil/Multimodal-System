#!/bin/bash

echo "🎨 Starting AI Multi-Modal Analysis Frontend..."
echo ""

# Navigate to frontend directory
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies (this may take a few minutes)..."
    npm install
    echo "✅ Dependencies installed!"
fi

echo ""
echo "🎯 Starting React development server on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the development server
npm run dev
