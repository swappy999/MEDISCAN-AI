import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config();

function getCxRServiceUrl() {
  return process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';
}

function getMriServiceUrl() {
  return process.env.MRI_SERVICE_URL || 'http://127.0.0.1:8001';
}

async function checkServiceHealth(baseUrl, serviceLabel) {
  try {
    const response = await axios.get(`${baseUrl}/health`, {
      timeout: 5000,
    });
    return {
      ...response.data,
      service_url: baseUrl,
    };
  } catch (error) {
    const errorMsg =
      error.response?.data?.detail ||
      error.message ||
      `Unable to connect to ${serviceLabel}`;
    throw new Error(`${serviceLabel} health check failed at ${baseUrl}: ${errorMsg}`);
  }
}

async function postImageFile(endpointUrl, imageFilePath, timeoutMs, fallbackError) {
  if (!fs.existsSync(imageFilePath)) {
    throw new Error(`Image file not found at path: ${imageFilePath}`);
  }

  const formData = new FormData();
  const filename = path.basename(imageFilePath);
  formData.append('file', fs.createReadStream(imageFilePath), { filename });

  try {
    const response = await axios.post(endpointUrl, formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: timeoutMs,
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      const resData = error.response.data || {};
      const status = error.response.status;
      const errorMsg =
        resData.message || resData.error || resData.detail || error.response.statusText || fallbackError;
      const customErr = new Error(
        typeof errorMsg === 'object' ? JSON.stringify(errorMsg) : errorMsg
      );
      customErr.statusCode = status;
      customErr.validation = resData.validation || (status === 422 ? resData : null);
      customErr.data = resData;
      throw customErr;
    }
    if (error.request) {
      const netErr = new Error(
        `ML service unreachable at ${endpointUrl}. Please ensure the FastAPI service is running.`
      );
      netErr.statusCode = 503;
      throw netErr;
    }
    throw new Error(`Request setup error: ${error.message}`);
  }
}

export async function checkHealth() {
  return checkServiceHealth(getCxRServiceUrl(), 'Chest X-ray ML service');
}

export async function validateImage(imageFilePath) {
  return postImageFile(
    `${getCxRServiceUrl()}/validate`,
    imageFilePath,
    30000,
    'Chest X-ray validation rejected the image'
  );
}

export async function predict(imageFilePath) {
  return postImageFile(
    `${getCxRServiceUrl()}/predict`,
    imageFilePath,
    60000,
    'Chest X-ray prediction failed'
  );
}

export async function checkMriHealth() {
  return checkServiceHealth(getMriServiceUrl(), 'Brain MRI ML service');
}

export async function validateMriImage(imageFilePath) {
  return postImageFile(
    `${getMriServiceUrl()}/validate`,
    imageFilePath,
    30000,
    'Brain MRI validation rejected the image'
  );
}

export async function predictMri(imageFilePath) {
  return postImageFile(
    `${getMriServiceUrl()}/predict`,
    imageFilePath,
    60000,
    'Brain MRI prediction failed'
  );
}
