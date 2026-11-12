import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import LoadingSpinner from './LoadingSpinner'

export default function AnalysisUploadView({
  title,
  subtitle,
  dropzoneLabel,
  pipelineLabel,
  spinnerTitle,
  spinnerSubtitle,
  submitLabel,
  classifierName,
  validationThresholdLabel,
  clinicalGuidance,
  rejectionFallbackMessage,
  rejectionFallbackReason,
  rejectionDefaultType,
  analyzeFn,
}) {
  const navigate = useNavigate()
  const fileInputRef = useRef(null)

  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errorMessage, setErrorMessage] = useState(null)
  const [validationError, setValidationError] = useState(null)

  const handleFileChange = (selectedFile) => {
    setErrorMessage(null)
    setValidationError(null)
    if (!selectedFile) return

    const validTypes = ['image/jpeg', 'image/png', 'image/jpg']
    if (!validTypes.includes(selectedFile.type)) {
      setErrorMessage(
        `Invalid file type (${selectedFile.type || 'unknown'}). Please select a JPEG or PNG image.`
      )
      return
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setErrorMessage('File size exceeds the 10MB limit.')
      return
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setFile(selectedFile)
    setPreviewUrl(URL.createObjectURL(selectedFile))
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0])
    }
  }

  const handleClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setFile(null)
    setPreviewUrl(null)
    setErrorMessage(null)
    setValidationError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file || isSubmitting) return

    setIsSubmitting(true)
    setErrorMessage(null)
    setValidationError(null)

    try {
      const formData = new FormData()
      formData.append('image', file)

      const result = await analyzeFn(formData)
      if (result && result.id) {
        navigate(`/analysis/${result.id}`)
      } else {
        throw new Error('Analysis succeeded but returned an unexpected response format.')
      }
    } catch (err) {
      if (err.validation || err.status === 422) {
        setValidationError({
          message: err.message || rejectionFallbackMessage,
          imageType: err.validation?.image_type || rejectionDefaultType,
          status: err.validation?.validation_status || 'REJECTED',
          reason: err.validation?.reason || rejectionFallbackReason,
          score: err.validation?.score != null ? err.validation.score : null,
          distance: err.validation?.distance != null ? err.validation.distance : null,
        })
      } else {
        setErrorMessage(err.message || 'An error occurred during image analysis.')
      }
      setIsSubmitting(false)
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
  }

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <div style={{ marginBottom: '28px' }}>
        <h1 style={{ fontSize: '26px', color: 'var(--navy)', marginBottom: '4px' }}>{title}</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>{subtitle}</p>
      </div>

      {errorMessage && (
        <div
          style={{
            padding: '14px 18px',
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            color: 'var(--danger)',
            borderRadius: '6px',
            marginBottom: '20px',
            fontSize: '13.5px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>{errorMessage}</span>
        </div>
      )}

      {validationError && (
        <div
          style={{
            padding: '24px',
            backgroundColor: '#fff1f2',
            border: '1.5px solid #f43f5e',
            borderRadius: '8px',
            marginBottom: '24px',
            boxShadow: '0 2px 4px rgba(244, 63, 94, 0.08)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '14px' }}>
            <div
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                backgroundColor: '#ffe4e6',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#e11d48',
                flexShrink: 0,
              }}
            >
              <svg
                width="22"
                height="22"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="15" y1="9" x2="9" y2="15"></line>
                <line x1="9" y1="9" x2="15" y2="15"></line>
              </svg>
            </div>

            <div style={{ flex: 1 }}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  flexWrap: 'wrap',
                  marginBottom: '6px',
                }}
              >
                <span
                  style={{
                    backgroundColor: '#e11d48',
                    color: '#ffffff',
                    fontSize: '11px',
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: '4px',
                    letterSpacing: '0.04em',
                    textTransform: 'uppercase',
                  }}
                >
                  {validationError.status}: {validationError.imageType}
                </span>
                <span style={{ fontSize: '12px', color: '#881337', fontWeight: 600 }}>
                  Pre-Inference Validation Gating
                </span>
              </div>

              <h2
                style={{
                  fontSize: '16px',
                  fontWeight: 700,
                  color: '#9f1239',
                  margin: '4px 0 8px 0',
                  lineHeight: 1.3,
                }}
              >
                {validationError.message}
              </h2>

              <p
                style={{
                  fontSize: '13.5px',
                  color: '#4c0519',
                  margin: '0 0 12px 0',
                  lineHeight: 1.5,
                }}
              >
                {validationError.reason}
              </p>

              <div
                style={{
                  background: '#ffffff',
                  border: '1px solid #fecdd3',
                  borderRadius: '6px',
                  padding: '12px 14px',
                  fontSize: '12.5px',
                  display: 'grid',
                  gridTemplateColumns: 'auto 1fr',
                  gap: '6px 14px',
                  color: '#4c0519',
                  marginBottom: '14px',
                }}
              >
                <span style={{ fontWeight: 600, color: '#9f1239' }}>Safety Action:</span>
                <span>
                  {classifierName} was <strong>NOT called</strong>. No heatmaps, scores, or reports
                  were generated.
                </span>

                {validationError.score != null && (
                  <>
                    <span style={{ fontWeight: 600, color: '#9f1239' }}>Similarity Metric:</span>
                    <span style={{ fontFamily: 'monospace' }}>
                      {(validationError.score * 100).toFixed(1)}% ({validationThresholdLabel})
                    </span>
                  </>
                )}

                <span style={{ fontWeight: 600, color: '#9f1239' }}>Clinical Guidance:</span>
                <span>{clinicalGuidance}</span>
              </div>

              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  type="button"
                  onClick={handleClear}
                  style={{
                    backgroundColor: '#e11d48',
                    color: '#ffffff',
                    border: 'none',
                    padding: '8px 16px',
                    borderRadius: '5px',
                    fontSize: '13px',
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                >
                  Upload Another Image
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {isSubmitting ? (
        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            padding: '60px 24px',
            textAlign: 'center',
            boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
          }}
        >
          <LoadingSpinner size={44} message={spinnerTitle} />
          <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginTop: '8px' }}>
            {spinnerSubtitle}
          </p>
        </div>
      ) : (
        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            padding: '32px',
            boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            style={{ display: 'none' }}
            onChange={(e) => handleFileChange(e.target.files[0])}
          />

          {!file ? (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
              style={{
                border: `2px dashed ${isDragging ? 'var(--cyan)' : 'var(--border)'}`,
                borderRadius: '8px',
                padding: '48px 24px',
                textAlign: 'center',
                backgroundColor: isDragging ? '#f0f9ff' : '#f8fafc',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              <div
                style={{
                  width: '56px',
                  height: '56px',
                  margin: '0 auto 16px auto',
                  borderRadius: '50%',
                  backgroundColor: '#ffffff',
                  border: '1px solid var(--border)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--navy)',
                }}
              >
                <svg
                  width="26"
                  height="26"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
              </div>

              <div
                style={{
                  fontSize: '15px',
                  fontWeight: 600,
                  color: 'var(--navy)',
                  marginBottom: '4px',
                }}
              >
                {dropzoneLabel}
              </div>
              <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                Supports standard formats: JPEG, PNG (up to 10MB)
              </div>
            </div>
          ) : (
            <div>
              <div
                style={{
                  display: 'flex',
                  gap: '24px',
                  alignItems: 'flex-start',
                  backgroundColor: '#f8fafc',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  padding: '20px',
                }}
              >
                <div
                  style={{
                    width: '180px',
                    height: '180px',
                    backgroundColor: '#000000',
                    borderRadius: '6px',
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                  }}
                >
                  <img
                    src={previewUrl}
                    alt="Scan preview"
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  />
                </div>

                <div
                  style={{
                    flex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    minHeight: '180px',
                  }}
                >
                  <div>
                    <h3
                      style={{
                        fontSize: '15px',
                        color: 'var(--navy)',
                        marginBottom: '8px',
                        wordBreak: 'break-all',
                      }}
                    >
                      {file.name}
                    </h3>
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'auto 1fr',
                        gap: '6px 16px',
                        fontSize: '13px',
                      }}
                    >
                      <span style={{ color: 'var(--text-muted)' }}>MIME Type:</span>
                      <span style={{ fontWeight: 500, fontFamily: 'monospace' }}>
                        {file.type || 'image/jpeg'}
                      </span>

                      <span style={{ color: 'var(--text-muted)' }}>File Size:</span>
                      <span style={{ fontWeight: 500 }}>{formatFileSize(file.size)}</span>

                      <span style={{ color: 'var(--text-muted)' }}>Pipeline:</span>
                      <span style={{ color: 'var(--navy)', fontWeight: 600 }}>{pipelineLabel}</span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
                    <button
                      type="button"
                      onClick={handleClear}
                      style={{
                        background: 'transparent',
                        color: 'var(--danger)',
                        border: '1px solid #fecaca',
                        padding: '6px 14px',
                        borderRadius: '6px',
                        fontSize: '13px',
                        fontWeight: 500,
                        cursor: 'pointer',
                      }}
                    >
                      Remove File
                    </button>
                    <button
                      type="button"
                      onClick={() => fileInputRef.current && fileInputRef.current.click()}
                      style={{
                        background: '#ffffff',
                        color: 'var(--text)',
                        border: '1px solid var(--border)',
                        padding: '6px 14px',
                        borderRadius: '6px',
                        fontSize: '13px',
                        fontWeight: 500,
                        cursor: 'pointer',
                      }}
                    >
                      Choose Different Image
                    </button>
                  </div>
                </div>
              </div>

              <div
                style={{
                  marginTop: '24px',
                  display: 'flex',
                  justifyContent: 'flex-end',
                  gap: '12px',
                }}
              >
                <button
                  type="button"
                  onClick={handleClear}
                  style={{
                    background: 'transparent',
                    border: '1px solid var(--border)',
                    color: 'var(--text)',
                    padding: '10px 18px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontWeight: 500,
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={!file || isSubmitting}
                  style={{
                    backgroundColor: 'var(--navy)',
                    color: '#ffffff',
                    border: 'none',
                    padding: '10px 24px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontWeight: 600,
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                  }}
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                  </svg>
                  {submitLabel}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

