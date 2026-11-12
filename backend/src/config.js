import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
export const backendRoot = path.resolve(__dirname, '..');

dotenv.config({ path: path.join(backendRoot, '.env') });

export const uploadsDir = path.isAbsolute(process.env.UPLOADS_DIR || '')
  ? process.env.UPLOADS_DIR
  : path.resolve(backendRoot, process.env.UPLOADS_DIR || './uploads');

export const resultsDir = path.isAbsolute(process.env.RESULTS_DIR || '')
  ? process.env.RESULTS_DIR
  : path.resolve(backendRoot, process.env.RESULTS_DIR || './results');

export const dbPath = path.isAbsolute(process.env.DB_PATH || '')
  ? process.env.DB_PATH
  : path.resolve(backendRoot, process.env.DB_PATH || './mediscan.db');

