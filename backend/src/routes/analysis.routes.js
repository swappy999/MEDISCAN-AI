import { Router } from 'express';
import path from 'path';
import fs from 'fs';
import { uploadSingle } from '../middleware/upload.middleware.js';
import * as controller from '../controllers/analysis.controller.js';
import * as mlService from '../services/ml.service.js';
import { uploadsDir, resultsDir } from '../config.js';

const router = Router();

// GET /api/health
router.get('/health', async (req, res) => {
  const [cxrOutcome, mriOutcome] = await Promise.allSettled([
    mlService.checkHealth(),
    mlService.checkMriHealth(),
  ]);

  const mlServiceStatus =
    cxrOutcome.status === 'fulfilled'
      ? cxrOutcome.value
      : { status: 'unavailable', error: cxrOutcome.reason?.message || 'Unavailable' };

  const mriServiceStatus =
    mriOutcome.status === 'fulfilled'
      ? mriOutcome.value
      : { status: 'unavailable', error: mriOutcome.reason?.message || 'Unavailable' };

  return res.status(200).json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    mlService: mlServiceStatus,
    mriService: mriServiceStatus,
  });
});

// Chest X-Ray endpoints (preserved)
router.post('/validate', uploadSingle, controller.validateOnly);
router.post('/analysis', uploadSingle, controller.createAnalysis);

// Brain MRI endpoints (isolated pipeline)
router.post('/mri/validate', uploadSingle, controller.validateMriOnly);
router.post('/mri/analysis', uploadSingle, controller.createMriAnalysis);

// Shared analysis retrieval, reporting & asset endpoints
router.get('/analysis', controller.getAnalyses);
router.get('/analysis/:id', controller.getAnalysis);
router.get('/analysis/:id/report', controller.downloadReport);
router.delete('/analysis/:id', controller.deleteAnalysis);

// GET /api/results/:filename (with strict path-traversal prevention)
router.get('/results/:filename', (req, res) => {
  const filename = req.params.filename;

  if (
    !filename ||
    filename.includes('..') ||
    filename.includes('/') ||
    filename.includes('\\') ||
    path.basename(filename) !== filename
  ) {
    return res.status(400).json({ error: 'Invalid filename' });
  }

  let targetPath = path.join(resultsDir, filename);
  if (!fs.existsSync(targetPath)) {
    const uploadPath = path.join(uploadsDir, filename);
    if (fs.existsSync(uploadPath)) {
      targetPath = uploadPath;
    } else {
      return res.status(404).json({ error: 'File not found' });
    }
  }

  const resolved = path.resolve(targetPath);
  const isAllowed = [resultsDir, uploadsDir].some(
    (dir) => resolved.startsWith(dir + path.sep) || resolved === dir
  );

  if (!isAllowed) {
    return res.status(403).json({ error: 'Access denied' });
  }

  return res.sendFile(resolved);
});

export default router;
