#!/bin/bash

# Set environment variables
export MONGODB_URI="mongodb+srv://kaushik2003singh:Fg3yrUlzZPaH9R7y@complaintapp.shaxxqw.mongodb.net/facevault?retryWrites=true&w=majority&appName=ComplaintApp"
export MONGODB_DB_NAME="facevault"
export JWT_SECRET_KEY="your-secure-secret-key-here"
export JWT_ALGORITHM="HS256"
export ACCESS_TOKEN_EXPIRE_MINUTES="10080"
export RAILWAY_ENVIRONMENT="development"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies
pip install -r requirements.txt

# Start the application
python main.py

# Start the backend server
cd backend
./start.sh &
BACKEND_PID=$!

# Start the frontend development server
npm run dev &
FRONTEND_PID=$!

# Function to handle script termination
cleanup() {
    echo "Shutting down servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit 0
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

echo "FaceVault is running!"
echo "Backend server: http://localhost:8000"
echo "Frontend server: http://localhost:3000"
echo "Press Ctrl+C to stop both servers."

# Wait for both processes
wait
