import express from 'express';
import cors from 'cors';
import multer from 'multer';
import dotenv from 'dotenv';
import analysisRoutes from './src/routes/analysis.routes.js';

dotenv.config();

const app = express();

const frontendUrl = process.env.FRONTEND_URL || 'http://localhost:5173';

// CORS configuration
app.use(
  cors({
    origin: [frontendUrl, 'http://localhost:5173', 'http://127.0.0.1:5173'],
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
  })
);

// Standard body parsing middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Primary API routes
app.use('/api', analysisRoutes);

// Fallback for /results/:filename if accessed directly without /api prefix
app.use('/results', (req, res, next) => {
  req.url = `/results${req.url}`;
  analysisRoutes(req, res, next);
});

// Centralized error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);

  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File size exceeds limit (10MB maximum)' });
    }
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  }

  if (err.message && err.message.startsWith('Invalid file type')) {
    return res.status(400).json({ error: err.message });
  }

  const statusCode = err.status || 500;
  return res.status(statusCode).json({
    error: err.message || 'Internal Server Error',
  });
});

export default app;
