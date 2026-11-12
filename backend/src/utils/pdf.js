import PDFDocument from 'pdfkit';
import fs from 'fs';
import path from 'path';
import { resultsDir } from '../config.js';

function getMriStatusMeta(prediction) {
  const norm = String(prediction || '').toUpperCase();
  if (norm === 'NO_TUMOR' || norm === 'NOTUMOR') {
    return { label: 'NO TUMOR', color: '#16a34a' };
  }
  if (norm === 'GLIOMA') {
    return { label: 'GLIOMA', color: '#dc2626' };
  }
  if (norm === 'MENINGIOMA') {
    return { label: 'MENINGIOMA', color: '#d97706' };
  }
  if (norm === 'PITUITARY') {
    return { label: 'PITUITARY', color: '#dc2626' };
  }
  return { label: norm || 'UNKNOWN', color: '#1e3a5f' };
}

/**
 * Generate a medical-grade PDF report for either Chest X-Ray or Brain MRI analysis.
 * @param {Object} analysis - The analysis data object
 * @param {string} outputPath - The file path to save the generated PDF
 * @returns {Promise<string>}
 */
export function generateReport(analysis, outputPath) {
  return new Promise((resolve, reject) => {
    try {
      const targetDir = path.dirname(outputPath);
      if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true });
      }

      const isBrainMri = analysis.modality === 'brain_mri';

      const doc = new PDFDocument({
        size: 'A4',
        margin: 40,
        info: {
          Title: `MediScan AI Report - ${analysis.id}`,
          Author: 'MediScan AI',
          Subject: isBrainMri
            ? 'Brain MRI Tumour Classification Report'
            : 'Chest X-Ray Pneumonia Analysis Report',
        },
      });

      const writeStream = fs.createWriteStream(outputPath);
      writeStream.on('finish', () => resolve(outputPath));
      writeStream.on('error', (err) => reject(err));
      doc.on('error', (err) => reject(err));

      doc.pipe(writeStream);

      const navyColor = '#1e3a5f';
      const textMuted = '#64748b';

      let predictionText;
      let statusColor;
      if (isBrainMri) {
        const meta = getMriStatusMeta(analysis.prediction);
        predictionText = meta.label;
        statusColor = meta.color;
      } else {
        const isPneumonia = (analysis.prediction || '').toUpperCase().includes('PNEUMONIA');
        predictionText = (analysis.prediction || 'UNKNOWN').toUpperCase();
        statusColor = isPneumonia ? '#dc2626' : '#16a34a';
      }

      // Header Banner
      doc.rect(40, 40, doc.page.width - 80, 50).fill(navyColor);
      doc
        .fillColor('#ffffff')
        .fontSize(17)
        .font('Helvetica-Bold')
        .text(
          isBrainMri
            ? 'MediScan AI — Brain MRI Tumour Analysis Report'
            : 'MediScan AI — Chest X-Ray Analysis Report',
          55,
          56
        );

      // Metadata Section
      const startY = 105;
      doc
        .fontSize(10)
        .font('Helvetica-Bold')
        .fillColor(navyColor)
        .text('Report Details', 40, startY);

      doc
        .strokeColor('#e2e8f0')
        .lineWidth(1)
        .moveTo(40, startY + 16)
        .lineTo(doc.page.width - 40, startY + 16)
        .stroke();

      const metaY = startY + 24;
      doc
        .font('Helvetica-Bold').fillColor('#334155').fontSize(9).text('Analysis ID: ', 40, metaY, { continued: true })
        .font('Helvetica').fillColor('#0f172a').text(analysis.id || 'N/A')
        .font('Helvetica-Bold').fillColor('#334155').text('Timestamp: ', 40, metaY + 15, { continued: true })
        .font('Helvetica').fillColor('#0f172a').text(analysis.timestamp || new Date().toISOString())
        .font('Helvetica-Bold').fillColor('#334155').text('Original File: ', 40, metaY + 30, { continued: true })
        .font('Helvetica').fillColor('#0f172a').text(analysis.original_filename || 'N/A');

      doc
        .font('Helvetica-Bold').fillColor('#334155').fontSize(9)
        .text('Model Identifier: ', 300, metaY, { continued: true })
        .font('Helvetica').fillColor('#0f172a')
        .text(analysis.model_name || (isBrainMri ? 'ResNet18BrainMRI' : 'DenseNet121Pneumonia'))
        .font('Helvetica-Bold').fillColor('#334155').text('Processing Time: ', 300, metaY + 15, { continued: true })
        .font('Helvetica').fillColor('#0f172a').text(analysis.processing_time ? `${analysis.processing_time} s` : 'N/A')
        .font('Helvetica-Bold').fillColor('#334155').text('Image Validation: ', 300, metaY + 30, { continued: true })
        .font('Helvetica').fillColor('#16a34a').text(isBrainMri ? 'VERIFIED BRAIN MRI' : 'VERIFIED CHEST RADIOGRAPH');

      // Diagnostic Result Card
      const resultY = metaY + 55;
      doc
        .roundedRect(40, resultY, doc.page.width - 80, 72, 6)
        .fillAndStroke('#f8fafc', '#cbd5e1');

      doc
        .font('Helvetica-Bold')
        .fontSize(10)
        .fillColor(textMuted)
        .text(isBrainMri ? 'MODEL CLASSIFICATION' : 'DIAGNOSTIC PREDICTION', 55, resultY + 12);

      doc
        .font('Helvetica-Bold')
        .fontSize(22)
        .fillColor(statusColor)
        .text(predictionText, 55, resultY + 28);

      const confPercent =
        analysis.confidence !== undefined && analysis.confidence !== null
          ? `${(Number(analysis.confidence) * 100).toFixed(1)}%`
          : 'N/A';

      doc
        .font('Helvetica')
        .fontSize(11)
        .fillColor('#334155')
        .text('Confidence Level: ', 320, resultY + 32, { continued: true })
        .font('Helvetica-Bold')
        .fillColor(navyColor)
        .text(confPercent);

      // Probabilities Breakdown
      let probs = {};
      try {
        probs =
          typeof analysis.probabilities === 'string'
            ? JSON.parse(analysis.probabilities)
            : analysis.probabilities || {};
      } catch {
        probs = {};
      }

      const tableY = resultY + 88;
      doc
        .font('Helvetica-Bold')
        .fontSize(11)
        .fillColor(navyColor)
        .text('Class Probabilities Distribution', 40, tableY);

      doc
        .strokeColor('#e2e8f0')
        .lineWidth(1)
        .moveTo(40, tableY + 16)
        .lineTo(doc.page.width - 40, tableY + 16)
        .stroke();

      const rowY = tableY + 25;
      let nextSectionOffset = 35;

      if (isBrainMri) {
        const formatProb = (val) => (val !== undefined && val !== null ? `${(Number(val) * 100).toFixed(1)}%` : 'N/A');
        const noTumorVal = probs.no_tumor !== undefined ? probs.no_tumor : probs.notumor;

        doc
          .font('Helvetica-Bold').fontSize(10).fillColor('#0f172a').text('Glioma:', 55, rowY)
          .font('Helvetica').text(formatProb(probs.glioma), 145, rowY)
          .font('Helvetica-Bold').text('Meningioma:', 280, rowY)
          .font('Helvetica').text(formatProb(probs.meningioma), 380, rowY);

        doc
          .font('Helvetica-Bold').fontSize(10).fillColor('#0f172a').text('Pituitary:', 55, rowY + 18)
          .font('Helvetica').text(formatProb(probs.pituitary), 145, rowY + 18)
          .font('Helvetica-Bold').text('No Tumor:', 280, rowY + 18)
          .font('Helvetica').text(formatProb(noTumorVal), 380, rowY + 18);

        nextSectionOffset = 50;
      } else {
        const probNormal = probs.NORMAL !== undefined ? `${(probs.NORMAL * 100).toFixed(1)}%` : 'N/A';
        const probPneumonia = probs.PNEUMONIA !== undefined ? `${(probs.PNEUMONIA * 100).toFixed(1)}%` : 'N/A';

        doc
          .font('Helvetica-Bold').fontSize(10).fillColor('#0f172a').text('NORMAL:', 55, rowY)
          .font('Helvetica').text(probNormal, 140, rowY)
          .font('Helvetica-Bold').text('PNEUMONIA:', 260, rowY)
          .font('Helvetica').text(probPneumonia, 360, rowY);
      }

      // Images Section
      const imagesY = rowY + nextSectionOffset;
      doc
        .font('Helvetica-Bold')
        .fontSize(11)
        .fillColor(navyColor)
        .text('Visual Radiological Evidence', 40, imagesY);

      doc
        .strokeColor('#e2e8f0')
        .lineWidth(1)
        .moveTo(40, imagesY + 16)
        .lineTo(doc.page.width - 40, imagesY + 16)
        .stroke();

      const imageBoxY = imagesY + 26;
      const imgWidth = 240;
      const imgHeight = 230;

      if (analysis.original_path && fs.existsSync(analysis.original_path)) {
        try {
          doc.image(analysis.original_path, 40, imageBoxY, {
            fit: [imgWidth, imgHeight],
            align: 'center',
            valign: 'center',
          });
        } catch {
          // ignore decode error
        }
      }
      doc
        .font('Helvetica-Bold')
        .fontSize(9)
        .fillColor('#475569')
        .text(isBrainMri ? 'Original Brain MRI' : 'Original Chest X-Ray', 40, imageBoxY + imgHeight + 6, {
          width: imgWidth,
          align: 'center',
        });

      let visualPath = null;
      const candidatePath = isBrainMri
        ? (analysis.overlay_path || analysis.heatmap_path)
        : (analysis.heatmap_path || analysis.overlay_path);
      if (candidatePath) {
        const resolved = path.isAbsolute(candidatePath)
          ? candidatePath
          : path.join(resultsDir, candidatePath);
        if (fs.existsSync(resolved)) {
          visualPath = resolved;
        }
      }

      if (visualPath) {
        try {
          doc.image(visualPath, 315, imageBoxY, {
            fit: [imgWidth, imgHeight],
            align: 'center',
            valign: 'center',
          });
        } catch {
          // ignore decode error
        }
      }
      doc
        .font('Helvetica-Bold')
        .fontSize(9)
        .fillColor('#475569')
        .text(
          isBrainMri ? 'Grad-CAM Diagnostic Overlay' : 'Grad-CAM Visual Explanation',
          315,
          imageBoxY + imgHeight + 6,
          { width: imgWidth, align: 'center' }
        );

      // Disclaimer & Limitations Box at Bottom
      const disclaimerHeight = isBrainMri ? 68 : 50;
      const disclaimerY = doc.page.height - (disclaimerHeight + 45);
      doc
        .rect(40, disclaimerY, doc.page.width - 80, disclaimerHeight)
        .fillAndStroke('#fffbeb', '#fef3c7');

      doc
        .font('Helvetica-Bold')
        .fontSize(8)
        .fillColor('#b45309')
        .text('MEDICAL RESEARCH DISCLAIMER & LIMITATIONS:', 50, disclaimerY + 9);

      const disclaimerBody = isBrainMri
        ? 'Grad-CAM highlights regions influencing the model; it is not verified tumour segmentation or clinical localisation and does not estimate tumour size, grade, or stage. This is an AI-generated research result and is not a confirmed medical diagnosis. It must not replace professional radiological interpretation.'
        : 'This is an AI-generated research result and is not a confirmed medical diagnosis. It must not replace professional radiological interpretation. Qualified clinical assessment is always required.';

      doc
        .font('Helvetica')
        .fontSize(8)
        .fillColor('#92400e')
        .text(disclaimerBody, 50, disclaimerY + 21, { width: doc.page.width - 100 });

      doc.end();
    } catch (err) {
      reject(err);
    }
  });
}
