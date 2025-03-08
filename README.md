<<<<<<< HEAD
# ESKAPE Analysis Application

A web application for analyzing microscopic urine sample images to detect ESKAPE category bacteria.
=======
# EKAPE Analysis Application

A web application for analyzing microscopic urine sample images to detect EKAPE category bacteria.
>>>>>>> 14e0549d229dde9587a586ca051c49e5cc18e85e

## Prerequisites

- Node.js (v18 or higher)
- MongoDB installed locally or MongoDB Atlas account
- Git

## Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following content:
```
MONGODB_URI=your_mongodb_connection_string
```

Replace `your_mongodb_connection_string` with either:
<<<<<<< HEAD
- Local MongoDB: `mongodb://localhost:27017/eskape_analysis`
- MongoDB Atlas: `mongodb+srv://<username>:<password>@<cluster>.mongodb.net/eskape_analysis`
=======
- Local MongoDB: `mongodb://localhost:27017/ekape_analysis`
- MongoDB Atlas: `mongodb+srv://<username>:<password>@<cluster>.mongodb.net/ekape_analysis`
>>>>>>> 14e0549d229dde9587a586ca051c49e5cc18e85e

## Running the Application

1. Start the development server:
```bash
npm run dev
```

This will start both the backend server and frontend development server.

2. Access the application:
- Open your browser and navigate to `http://localhost:5000`
- The API endpoints will be available at `http://localhost:5000/api`

## Features

- Upload and analyze microscopic urine sample images
<<<<<<< HEAD
- Detect ESKAPE category bacteria
=======
- Detect EKAPE category bacteria
>>>>>>> 14e0549d229dde9587a586ca051c49e5cc18e85e
- View analysis results with confidence scores
- Store and retrieve analysis history
- Responsive design for all devices

## API Endpoints

- `GET /api/health` - Check server and database status
- `POST /api/analyze` - Upload and analyze an image
- `GET /api/analyses` - Get all analysis results
- `GET /api/analysis/:id` - Get specific analysis result

## Technology Stack

- Frontend: React with Vite, TailwindCSS, shadcn/ui
- Backend: Express.js, Node.js
- Database: MongoDB with Mongoose
<<<<<<< HEAD
- Image Processing: Sharp
=======
- Image Processing: Sharp
>>>>>>> 14e0549d229dde9587a586ca051c49e5cc18e85e
