import dotenv from 'dotenv';
import app from './app.js';

dotenv.config();

const PORT = parseInt(process.env.PORT || '5000', 10);
const FRONTEND_URL = process.env.FRONTEND_URL || 'http://localhost:5173';
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';

app.listen(PORT, () => {
  console.log(`=================================================`);
  console.log(` MediScan AI Backend Service`);
  console.log(`=================================================`);
  console.log(` Listening on:      http://localhost:${PORT}`);
  console.log(` Health check:      http://localhost:${PORT}/api/health`);
  console.log(` Frontend URL:      ${FRONTEND_URL}`);
  console.log(` ML Service URL:    ${ML_SERVICE_URL}`);
  console.log(`=================================================`);
});
