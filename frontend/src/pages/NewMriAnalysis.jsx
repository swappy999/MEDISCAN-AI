import React from 'react'
import { api } from '../services/api'
import AnalysisUploadView from '../components/AnalysisUploadView'

export default function NewMriAnalysis() {
  return (
    <AnalysisUploadView
      title="New Brain MRI Tumour Analysis"
      subtitle="Upload a brain MRI scan (JPEG or PNG) for 4-class classification (Glioma, Meningioma, Pituitary, No Tumor)"
      dropzoneLabel="Click to browse or drag and drop your Brain MRI scan"
      pipelineLabel="Brain MRI Validation → ResNet-18 Tumour Classification"
      spinnerTitle="Validating & analyzing Brain MRI..."
      spinnerSubtitle="Running Brain MRI OOD validation, ResNet-18 4-class inference, and Grad-CAM activation mapping."
      submitLabel="Analyze Brain MRI"
      classifierName="Brain MRI tumour classifier"
      validationThresholdLabel="cosine similarity with calibrated Brain MRI reference distribution"
      clinicalGuidance="Please upload an authentic Brain MRI scan in JPG, JPEG, or PNG format. Chest X-rays, other body-part X-rays/MRIs, CT scans, photos, and reports are rejected."
      rejectionFallbackMessage="Image rejected by Brain MRI validation system."
      rejectionFallbackReason="The uploaded image does not match calibrated Brain MRI criteria."
      rejectionDefaultType="NON_BRAIN_MRI"
      analyzeFn={api.analyzeMri}
    />
  )
}

