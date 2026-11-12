import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import * as mlService from '../services/ml.service.js';
import * as db from '../db/database.js';
import { generateReport } from '../utils/pdf.js';
import { uploadsDir, resultsDir } from '../config.js';

const safeDirs = [uploadsDir, resultsDir];

function safeDeleteFile(filePath) {
  if (!filePath) return;
  try {
    const resolved = path.resolve(filePath);
    const isSafe = safeDirs.some(
      (dir) => resolved.startsWith(dir + path.sep) || resolved === dir
    );
    if (isSafe && fs.existsSync(resolved)) {
      fs.unlinkSync(resolved);
    }
  } catch (err) {
    console.error(`Failed to delete file ${filePath}:`, err.message);
  }
}

function parseAnalysisRecord(record) {
  if (!record) return null;
  let probabilities = null;
  if (record.probabilities) {
    try {
      probabilities =
        typeof record.probabilities === 'string'
          ? JSON.parse(record.probabilities)
          : record.probabilities;
    } catch {
      probabilities = record.probabilities;
    }
  }
  return {
    ...record,
    modality: record.modality || 'chest_xray',
    probabilities,
  };
}

async function handleCreateAnalysis(req, res, { modality, predictFn, defaultModelName }) {
  if (!req.file) {
    return res.status(400).json({ error: 'No image file uploaded' });
  }

  const uploadedFilePath = req.file.path;
  let mlResult;

  try {
    mlResult = await predictFn(uploadedFilePath);
  } catch (err) {
    // Fail closed: clean up uploaded temp file and never insert DB record or PDF report
    safeDeleteFile(uploadedFilePath);
    if (err.statusCode === 422) {
      return res.status(422).json({
        success: false,
        error: err.message,
        validation: err.validation || null,
      });
    }
    return res.status(err.statusCode || 502).json({
      success: false,
      error: `ML service error: ${err.message}`,
    });
  }

  const analysisId = uuidv4();
  const timestamp = new Date().toISOString();
  const reportFilename = `${analysisId}_report.pdf`;
  const reportFullPath = path.join(resultsDir, reportFilename);

  const newRecord = {
    id: analysisId,
    timestamp,
    modality,
    original_filename: req.file.originalname,
    original_path: uploadedFilePath,
    prediction: mlResult.prediction,
    confidence: mlResult.confidence,
    probabilities:
      typeof mlResult.probabilities === 'object'
        ? JSON.stringify(mlResult.probabilities)
        : mlResult.probabilities,
    heatmap_path: mlResult.heatmap_path || null,
    overlay_path: mlResult.overlay_path || null,
    report_path: reportFilename,
    model_name: mlResult.model_name || defaultModelName,
    processing_time: mlResult.processing_time || null,
    validation_status: mlResult.validation?.validation_status || 'ACCEPTED',
  };

  try {
    await generateReport(
      {
        ...newRecord,
        probabilities: mlResult.probabilities,
      },
      reportFullPath
    );
  } catch (pdfErr) {
    console.error('PDF report generation error:', pdfErr.message);
    if (!fs.existsSync(reportFullPath)) {
      newRecord.report_path = null;
    }
  }

  try {
    const savedRecord = db.insertAnalysis(newRecord);
    return res.status(201).json(parseAnalysisRecord(savedRecord));
  } catch (dbErr) {
    console.error('Database insert error:', dbErr.message);
    return res.status(500).json({ error: `Database error: ${dbErr.message}` });
  }
}

async function handleValidateOnly(req, res, validateFn) {
  if (!req.file) {
    return res.status(400).json({ error: 'No image file uploaded' });
  }

  const uploadedFilePath = req.file.path;
  try {
    const result = await validateFn(uploadedFilePath);
    safeDeleteFile(uploadedFilePath);
    return res.status(200).json(result);
  } catch (err) {
    safeDeleteFile(uploadedFilePath);
    const statusCode = err.statusCode || 422;
    return res.status(statusCode).json({
      success: false,
      error: err.message,
      validation: err.validation || null,
    });
  }
}

export async function createAnalysis(req, res) {
  return handleCreateAnalysis(req, res, {
    modality: 'chest_xray',
    predictFn: mlService.predict,
    defaultModelName: 'DenseNet121Pneumonia',
  });
}

export async function createMriAnalysis(req, res) {
  return handleCreateAnalysis(req, res, {
    modality: 'brain_mri',
    predictFn: mlService.predictMri,
    defaultModelName: 'ResNet18BrainMRI',
  });
}

export function getAnalyses(req, res) {
  try {
    const modalityFilter = req.query.modality || null;
    const records = db.getAllAnalyses(modalityFilter);
    const parsed = records.map(parseAnalysisRecord);
    return res.status(200).json(parsed);
  } catch (err) {
    return res.status(500).json({ error: `Failed to fetch analyses: ${err.message}` });
  }
}

export function getAnalysis(req, res) {
  try {
    const record = db.getAnalysisById(req.params.id);
    if (!record) {
      return res.status(404).json({ error: 'Analysis not found' });
    }
    return res.status(200).json(parseAnalysisRecord(record));
  } catch (err) {
    return res.status(500).json({ error: `Failed to retrieve analysis: ${err.message}` });
  }
}

export function downloadReport(req, res) {
  try {
    const record = db.getAnalysisById(req.params.id);
    if (!record) {
      return res.status(404).json({ error: 'Analysis not found' });
    }

    if (!record.report_path) {
      return res.status(404).json({ error: 'Report not available for this analysis' });
    }

    const reportPath = path.isAbsolute(record.report_path)
      ? record.report_path
      : path.join(resultsDir, record.report_path);

    if (!fs.existsSync(reportPath)) {
      return res.status(404).json({ error: 'Report file not found on disk' });
    }

    const prefix = record.modality === 'brain_mri' ? 'mediscan-mri-report' : 'mediscan-report';
    return res.download(reportPath, `${prefix}-${record.id}.pdf`);
  } catch (err) {
    return res.status(500).json({ error: `Download error: ${err.message}` });
  }
}

export function deleteAnalysis(req, res) {
  try {
    const record = db.getAnalysisById(req.params.id);
    if (!record) {
      return res.status(404).json({ error: 'Analysis not found' });
    }

    db.deleteAnalysis(req.params.id);

    if (record.original_path) {
      safeDeleteFile(record.original_path);
    }

    for (const field of ['heatmap_path', 'overlay_path', 'report_path']) {
      if (record[field]) {
        const target = path.isAbsolute(record[field])
          ? record[field]
          : path.join(resultsDir, record[field]);
        safeDeleteFile(target);
      }
    }

    return res.status(200).json({
      message: 'Analysis deleted successfully',
      id: req.params.id,
    });
  } catch (err) {
    return res.status(500).json({ error: `Delete failed: ${err.message}` });
  }
}

export async function validateOnly(req, res) {
  return handleValidateOnly(req, res, mlService.validateImage);
}

export async function validateMriOnly(req, res) {
  return handleValidateOnly(req, res, mlService.validateMriImage);
}
