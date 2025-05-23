#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
#!/bin/bash

# Start the backend server
cd backend
./start.sh &
BACKEND_PID=$!

# Start the frontend development server
cd ..
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
