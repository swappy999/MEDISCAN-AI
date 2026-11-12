import React from 'react'
import { api } from '../services/api'
import AnalysisUploadView from '../components/AnalysisUploadView'

export default function NewAnalysis() {
  return (
    <AnalysisUploadView
      title="New Chest X-Ray Analysis"
      subtitle="Upload a frontal chest X-ray image (DICOM export, JPEG, or PNG) for pneumonia screening"
      dropzoneLabel="Click to browse or drag and drop your chest X-ray"
      pipelineLabel="CXR Validation → Pneumonia Screening"
      spinnerTitle="Validating & analyzing X-ray..."
      spinnerSubtitle="Running chest X-ray distribution validation, DenseNet-121 neural network inference, and Grad-CAM activation mapping."
      submitLabel="Analyze X-ray"
      classifierName="Pneumonia classifier"
      validationThresholdLabel="match with reference CXR distribution (calibrated acceptance threshold ≥ 78.2%)"
      clinicalGuidance="Please upload an authentic frontal chest X-ray (PA or AP view) in JPG, JPEG, or PNG format."
      rejectionFallbackMessage="Image rejected by chest X-ray validation system."
      rejectionFallbackReason="The uploaded image does not match genuine frontal chest X-ray criteria."
      rejectionDefaultType="NON_CHEST"
      analyzeFn={api.analyze}
    />
  )
}
