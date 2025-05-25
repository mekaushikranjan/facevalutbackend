# Set environment variables
$env:MONGODB_URI = "mongodb+srv://kaushik2003singh:Fg3yrUlzZPaH9R7y@complaintapp.shaxxqw.mongodb.net/facevault?retryWrites=true&w=majority&appName=ComplaintApp"
$env:MONGODB_DB_NAME = "facevault"
$env:JWT_SECRET_KEY = "your-secure-secret-key-here"
$env:JWT_ALGORITHM = "HS256"
$env:ACCESS_TOKEN_EXPIRE_MINUTES = "10080"
$env:RAILWAY_ENVIRONMENT = "development"

# Activate virtual environment if it exists
if (Test-Path "venv") {
    .\venv\Scripts\Activate.ps1
}

# Install dependencies
pip install -r requirements.txt

# Start the application
python main.py 