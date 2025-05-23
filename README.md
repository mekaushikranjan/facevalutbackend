# FaceVault Backend

This is the Python backend for the FaceVault application, which handles:

1. User authentication and authorization
2. Image upload and storage
3. Face detection and recognition
4. Album and people management

## Setup

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Run the backend server:
\`\`\`bash
uvicorn main:app --reload
\`\`\`

The server will start on port 8000.

## API Endpoints

### Authentication
- `POST /token`: Get JWT token
- `POST /users`: Create a new user
- `GET /users/me`: Get current user

### Images
- `POST /images`: Upload an image
- `GET /images`: List all images
- `GET /images/{image_id}`: Get image details
- `GET /images/{image_id}/content`: Get image content

### People
- `POST /people`: Create a new person
- `GET /people`: List all people
- `GET /people/{person_id}`: Get person details
- `PUT /people/{person_id}`: Update a person
- `POST /people/merge`: Merge multiple people

### Albums
- `POST /albums`: Create a new album
- `GET /albums`: List all albums
- `GET /albums/{album_id}`: Get album details
- `PUT /albums/{album_id}`: Update an album
- `POST /albums/{album_id}/images`: Add images to an album

### Faces
- `GET /faces/{face_id}/content`: Get face image content

## Face Detection

The backend uses the `face_recognition` library to detect faces in uploaded images. It automatically:

1. Detects faces in uploaded images
2. Extracts face encodings
3. Matches faces with existing people
4. Creates new people for unmatched faces
5. Updates people and image metadata

## Database

For simplicity, this demo uses in-memory storage. In a production environment, you would use a proper database like MongoDB.
\`\`\`

Let's create a start script for the backend:
