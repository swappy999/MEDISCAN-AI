import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import { dbPath } from '../config.js';

// Ensure directory exists for database file
const dbDir = path.dirname(dbPath);
if (!fs.existsSync(dbDir)) {
  fs.mkdirSync(dbDir, { recursive: true });
}

export const db = new Database(dbPath);

// Enable WAL mode for better concurrency
db.pragma('journal_mode = WAL');

// Initialize database schema
db.exec(`
  CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    modality TEXT DEFAULT 'chest_xray',
    original_filename TEXT,
    original_path TEXT,
    prediction TEXT,
    confidence REAL,
    probabilities TEXT,
    heatmap_path TEXT,
    overlay_path TEXT,
    report_path TEXT,
    model_name TEXT,
    processing_time REAL,
    validation_status TEXT
  )
`);

// Additive migrations for pre-existing mediscan.db files
const existingCols = db
  .prepare('PRAGMA table_info(analyses)')
  .all()
  .map((col) => col.name);

if (!existingCols.includes('modality')) {
  db.exec("ALTER TABLE analyses ADD COLUMN modality TEXT DEFAULT 'chest_xray'");
  db.exec("UPDATE analyses SET modality = 'chest_xray' WHERE modality IS NULL");
}
if (!existingCols.includes('validation_status')) {
  db.exec('ALTER TABLE analyses ADD COLUMN validation_status TEXT');
}

export function insertAnalysis(record) {
  const stmt = db.prepare(`
    INSERT INTO analyses (
      id,
      timestamp,
      modality,
      original_filename,
      original_path,
      prediction,
      confidence,
      probabilities,
      heatmap_path,
      overlay_path,
      report_path,
      model_name,
      processing_time,
      validation_status
    ) VALUES (
      @id,
      @timestamp,
      @modality,
      @original_filename,
      @original_path,
      @prediction,
      @confidence,
      @probabilities,
      @heatmap_path,
      @overlay_path,
      @report_path,
      @model_name,
      @processing_time,
      @validation_status
    )
  `);

  stmt.run({
    id: record.id,
    timestamp: record.timestamp,
    modality: record.modality || 'chest_xray',
    original_filename: record.original_filename || null,
    original_path: record.original_path || null,
    prediction: record.prediction || null,
    confidence: record.confidence !== undefined ? record.confidence : null,
    probabilities:
      typeof record.probabilities === 'object'
        ? JSON.stringify(record.probabilities)
        : record.probabilities || null,
    heatmap_path: record.heatmap_path || null,
    overlay_path: record.overlay_path || null,
    report_path: record.report_path || null,
    model_name: record.model_name || null,
    processing_time:
      record.processing_time !== undefined ? record.processing_time : null,
    validation_status: record.validation_status || null,
  });

  return getAnalysisById(record.id);
}

export function getAllAnalyses(modality = null) {
  if (modality === 'chest_xray' || modality === 'brain_mri') {
    const stmt = db.prepare(
      "SELECT * FROM analyses WHERE COALESCE(modality, 'chest_xray') = ? ORDER BY timestamp DESC"
    );
    return stmt.all(modality);
  }
  const stmt = db.prepare('SELECT * FROM analyses ORDER BY timestamp DESC');
  return stmt.all();
}

export function getAnalysisById(id) {
  const stmt = db.prepare('SELECT * FROM analyses WHERE id = ?');
  return stmt.get(id);
}

export function deleteAnalysis(id) {
  const record = getAnalysisById(id);
  if (!record) {
    return null;
  }
  const stmt = db.prepare('DELETE FROM analyses WHERE id = ?');
  stmt.run(id);
  return record;
}
