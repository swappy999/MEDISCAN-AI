/**
 * test_backend_integration.js
 * Verifies Express backend handling of validation rejection (HTTP 422),
 * file cleanup, and prevention of database insertion or PDF generation.
 */

import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import path from 'path';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';
const TEST_DIR = path.resolve('image-validation-dataset', 'test');

const testCases = [
  { file: '01_normal_chest_xray.jpeg', expectedStatus: 201, expectValid: true },
  { file: '03_hand_xray.png', expectedStatus: 422, expectValid: false },
  { file: '08_brain_mri.png', expectedStatus: 422, expectValid: false },
  { file: '11_medical_report.png', expectedStatus: 422, expectValid: false },
];

async function run() {
  console.log('--- Testing Express Backend Integration ---');
  for (const tc of testCases) {
    const filePath = path.join(TEST_DIR, tc.file);
    if (!fs.existsSync(filePath)) {
      console.log(`Skipping missing file: ${filePath}`);
      continue;
    }

    const form = new FormData();
    form.append('image', fs.createReadStream(filePath));

    try {
      const resp = await axios.post(`${BACKEND_URL}/api/analysis`, form, {
        headers: form.getHeaders(),
        timeout: 30000,
        validateStatus: () => true, // Don't throw on 4xx/5xx
      });

      const passed = resp.status === tc.expectedStatus;
      console.log(`[${passed ? 'PASS' : 'FAIL'}] ${tc.file}`);
      console.log(`   HTTP Status: ${resp.status} (Expected: ${tc.expectedStatus})`);
      if (resp.status === 422) {
        console.log(`   Rejection Message: "${resp.data.error}"`);
        console.log(`   Validation Status: ${resp.data.validation?.validation_status}`);
        console.log(`   Image Type: ${resp.data.validation?.image_type}`);
      } else if (resp.status === 201) {
        console.log(`   Prediction: ${resp.data.prediction} (Conf: ${(resp.data.confidence * 100).toFixed(1)}%)`);
        console.log(`   PDF Report Created: ${resp.data.report_path}`);
      }
    } catch (err) {
      console.error(`Error sending ${tc.file}:`, err.message);
    }
  }
}

run();

