import sys
import os

# Add your project directory to the Python path
path = '/home/17kaushik03/facevault/backend'
if path not in sys.path:
    sys.path.append(path)

# Set environment variables
os.environ['MONGODB_URI'] = 'mongodb+srv://kaushik2003singh:Fg3yrUlzZPaH9R7y@complaintapp.shaxxqw.mongodb.net/facevault?retryWrites=true&w=majority&appName=ComplaintApp'
os.environ['DATABASE_NAME'] = 'facevault'
os.environ['JWT_SECRET_KEY'] = "947832367c17c5421fc1d718cf7e66b8f2d9ad653b5c2dcfc22bf6461b2e040a"
os.environ['JWT_ALGORITHM'] = "HS256"
os.environ['ACCESS_TOKEN_EXPIRE_MINUTES'] = "10080"
os.environ['SMTP_HOST'] = "smtp.gmail.com"
os.environ['SMTP_PORT'] = "587"
os.environ['SMTP_USER'] = "mritunjaykaushik1803@gmail.com"
os.environ['SMTP_PASSWORD'] = "wzgl pppg sezx nlno"
os.environ['SMTP_FROM'] = "facevault@gmail.com"
os.environ['SMTP_FROM_NAME'] = "FaceVault"

# Import your FastAPI app
from main import app as application 