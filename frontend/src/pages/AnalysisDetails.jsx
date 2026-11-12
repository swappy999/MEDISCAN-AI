import React, { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api } from '../services/api'
import PredictionBadge from '../components/PredictionBadge'
import ModalityBadge from '../components/ModalityBadge'
import ImageModal from '../components/ImageModal'
import LoadingSpinner from '../components/LoadingSpinner'

export default function AnalysisDetails() {
  const { id } = useParams()
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeModalImage, setActiveModalImage] = useState(null) // { src, alt }

  useEffect(() => {
    let isMounted = true

    async function fetchAnalysis() {
      try {
        setLoading(true)
        setError(null)
        const data = await api.getAnalysis(id)
        if (isMounted) {
          setAnalysis(data)
        }
      } catch (err) {
        if (isMounted) {
          setError(err.message || 'Failed to load analysis details')
        }
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    fetchAnalysis()
    return () => { isMounted = false }
  }, [id])

  if (loading) {
    return <LoadingSpinner message="Loading analysis details..." />
  }

  if (error || !analysis) {
    return (
      <div style={{ maxWidth: '800px', margin: '40px auto', textAlign: 'center' }}>
        <div
          style={{
            padding: '24px',
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            color: 'var(--danger)',
            borderRadius: '8px',
            marginBottom: '20px'
          }}
        >
          <h3 style={{ fontSize: '16px', marginBottom: '8px' }}>Unable to Load Analysis</h3>
          <p style={{ fontSize: '13.5px' }}>{error || 'The requested analysis record was not found.'}</p>
        </div>
        <Link
          to="/history"
          style={{
            backgroundColor: 'var(--navy)',
            color: '#ffffff',
            padding: '10px 18px',
            borderRadius: '6px',
            fontWeight: 500,
            display: 'inline-block'
          }}
        >
          &larr; Back to History
        </Link>
      </div>
    )
  }

  // Parse probabilities safely
  let probs = analysis.probabilities
  if (typeof probs === 'string') {
    try {
      probs = JSON.parse(probs)
    } catch {
      probs = {}
    }
  }
  const modality = analysis.modality || 'chest_xray'
  const isMri = modality === 'brain_mri'

  const normProb = probs && probs.NORMAL != null ? Number(probs.NORMAL) : 0
  const pneuProb = probs && probs.PNEUMONIA != null ? Number(probs.PNEUMONIA) : 0
  const mriBars = [
    { label: 'Glioma', value: Number(probs?.glioma ?? probs?.GLIOMA ?? 0), color: 'var(--danger)' },
    { label: 'Meningioma', value: Number(probs?.meningioma ?? probs?.MENINGIOMA ?? 0), color: 'var(--warning)' },
    { label: 'Pituitary', value: Number(probs?.pituitary ?? probs?.PITUITARY ?? 0), color: '#ea580c' },
    { label: 'No Tumor', value: Number(probs?.no_tumor ?? probs?.notumor ?? probs?.NO_TUMOR ?? probs?.NOTUMOR ?? 0), color: 'var(--success)' }
  ]

  const confidencePct = analysis.confidence != null ? (analysis.confidence * 100).toFixed(1) : 'N/A'

  // Image URLs
  // Original could be served from results if saved there or filename
  const originalFilename = analysis.original_path
    ? analysis.original_path.split(/[/\\]/).pop()
    : analysis.original_filename
  const originalUrl = originalFilename ? api.resultUrl(originalFilename) : null
  const heatmapUrl = analysis.heatmap_path ? api.resultUrl(analysis.heatmap_path) : null
  const overlayUrl = analysis.overlay_path ? api.resultUrl(analysis.overlay_path) : null

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      {/* Navigation / Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <Link
            to="/history"
            style={{
              color: 'var(--text-muted)',
              fontSize: '13px',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              marginBottom: '6px'
            }}
          >
            &larr; Back to History
          </Link>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <h1 style={{ fontSize: '24px', color: 'var(--navy)' }}>
              Analysis Report: <span style={{ fontFamily: 'monospace', fontSize: '20px' }}>{analysis.id}</span>
            </h1>
            <ModalityBadge modality={modality} />
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <a
            href={api.reportUrl(analysis.id)}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              backgroundColor: '#ffffff',
              border: '1px solid var(--border)',
              color: 'var(--navy)',
              padding: '9px 16px',
              borderRadius: '6px',
              fontSize: '13.5px',
              fontWeight: 600,
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              cursor: 'pointer'
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            Download PDF Report
          </a>

          <Link
            to={isMri ? '/mri/new' : '/analysis/new'}
            style={{
              backgroundColor: 'var(--navy)',
              color: '#ffffff',
              padding: '9px 16px',
              borderRadius: '6px',
              fontSize: '13.5px',
              fontWeight: 600,
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            {isMri ? 'New Brain MRI' : 'New Chest X-Ray'}
          </Link>
        </div>
      </div>

      {/* Primary Diagnosis & Confidence Card */}
      <div
        style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          padding: '24px 28px',
          marginBottom: '24px',
          boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '32px'
        }}
      >
        {/* Left Side: Prediction & Confidence */}
        <div>
          <div style={{ color: 'var(--text-muted)', fontSize: '12.5px', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '8px' }}>
            Primary Classification
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '16px' }}>
            <PredictionBadge prediction={analysis.prediction} size="large" />
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px' }}>
              <span style={{ fontSize: '32px', fontWeight: 800, color: 'var(--navy)', lineHeight: 1 }}>
                {confidencePct}%
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>confidence</span>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 16px', fontSize: '13px', marginTop: '16px' }}>
            <span style={{ color: 'var(--text-muted)' }}>Modality:</span>
            <span style={{ fontWeight: 500 }}>{isMri ? 'Brain MRI' : 'Chest X-Ray'}</span>

            <span style={{ color: 'var(--text-muted)' }}>Image Filename:</span>
            <span style={{ fontWeight: 500 }}>{analysis.original_filename || 'Unknown'}</span>

            <span style={{ color: 'var(--text-muted)' }}>Model Architecture:</span>
            <span style={{ fontWeight: 500 }}>{analysis.model_name || (isMri ? 'ResNet-18 (Brain MRI)' : 'DenseNet121')}</span>

            <span style={{ color: 'var(--text-muted)' }}>Processing Time:</span>
            <span style={{ fontWeight: 500 }}>
              {analysis.processing_time != null ? `${Number(analysis.processing_time).toFixed(2)}s` : 'N/A'}
            </span>

            <span style={{ color: 'var(--text-muted)' }}>Timestamp:</span>
            <span style={{ fontWeight: 500 }}>
              {analysis.timestamp ? new Date(analysis.timestamp).toLocaleString() : 'N/A'}
            </span>
          </div>
        </div>

        {/* Right Side: Probabilities Bars */}
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '12.5px', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '14px' }}>
            Class Probabilities
          </div>

          {isMri ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {mriBars.map((bar) => (
                <div key={bar.label}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '5px' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text)' }}>{bar.label}</span>
                    <span style={{ fontWeight: 600, color: bar.color }}>{(bar.value * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ width: '100%', height: '9px', backgroundColor: '#e2e8f0', borderRadius: '5px', overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${Math.min(100, Math.max(0, bar.value * 100))}%`,
                        height: '100%',
                        backgroundColor: bar.color,
                        transition: 'width 0.4s ease'
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <>
              {/* Normal bar */}
              <div style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '6px' }}>
                  <span style={{ fontWeight: 600, color: 'var(--text)' }}>Normal</span>
                  <span style={{ fontWeight: 600, color: 'var(--success)' }}>{(normProb * 100).toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', height: '10px', backgroundColor: '#e2e8f0', borderRadius: '5px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${Math.min(100, Math.max(0, normProb * 100))}%`,
                      height: '100%',
                      backgroundColor: 'var(--success)',
                      transition: 'width 0.4s ease'
                    }}
                  />
                </div>
              </div>

              {/* Pneumonia bar */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '6px' }}>
                  <span style={{ fontWeight: 600, color: 'var(--text)' }}>Pneumonia</span>
                  <span style={{ fontWeight: 600, color: 'var(--danger)' }}>{(pneuProb * 100).toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', height: '10px', backgroundColor: '#e2e8f0', borderRadius: '5px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${Math.min(100, Math.max(0, pneuProb * 100))}%`,
                      height: '100%',
                      backgroundColor: 'var(--danger)',
                      transition: 'width 0.4s ease'
                    }}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Visualizations Card (3 images side by side) */}
      <div
        style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          padding: '24px',
          marginBottom: '24px',
          boxShadow: '0 1px 2px rgba(0,0,0,0.02)'
        }}
      >
        <div style={{ marginBottom: '16px' }}>
          <h2 style={{ fontSize: '16px', color: 'var(--navy)' }}>Radiological Visualizations</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '12.5px', marginTop: '2px' }}>
            Click any image to view in high resolution
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '20px' }}>
          {/* 1. Original Image */}
          <div
            onClick={() => originalUrl && setActiveModalImage({ src: originalUrl, alt: isMri ? 'Original Brain MRI' : 'Original Chest X-Ray' })}
            style={{
              border: '1px solid var(--border)',
              borderRadius: '6px',
              padding: '12px',
              backgroundColor: '#f8fafc',
              cursor: originalUrl ? 'pointer' : 'default',
              transition: 'border-color 0.15s ease'
            }}
          >
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--navy)', marginBottom: '8px' }}>
              {isMri ? '1. Original Brain MRI' : '1. Original X-Ray'}
            </div>
            <div
              style={{
                width: '100%',
                height: '240px',
                backgroundColor: '#000000',
                borderRadius: '4px',
                overflow: 'hidden',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              {originalUrl ? (
                <img
                  src={originalUrl}
                  alt={isMri ? 'Original Brain MRI' : 'Original Chest X-Ray'}
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  onError={(e) => {
                    e.currentTarget.style.display = 'none'
                    e.currentTarget.parentElement.innerHTML = '<span style="color:#94a3b8;font-size:12px;">Image unavailable</span>'
                  }}
                />
              ) : (
                <span style={{ color: '#94a3b8', fontSize: '12px' }}>Original not retained</span>
              )}
            </div>
          </div>

          {/* 2. Grad-CAM Heatmap */}
          <div
            onClick={() => heatmapUrl && setActiveModalImage({ src: heatmapUrl, alt: 'Grad-CAM Attention Heatmap' })}
            style={{
              border: '1px solid var(--border)',
              borderRadius: '6px',
              padding: '12px',
              backgroundColor: '#f8fafc',
              cursor: heatmapUrl ? 'pointer' : 'default',
              transition: 'border-color 0.15s ease'
            }}
          >
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--navy)', marginBottom: '8px' }}>
              2. Grad-CAM Heatmap
            </div>
            <div
              style={{
                width: '100%',
                height: '240px',
                backgroundColor: '#000000',
                borderRadius: '4px',
                overflow: 'hidden',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              {heatmapUrl ? (
                <img
                  src={heatmapUrl}
                  alt="Grad-CAM Attention Heatmap"
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  onError={(e) => {
                    e.currentTarget.style.display = 'none'
                    e.currentTarget.parentElement.innerHTML = '<span style="color:#94a3b8;font-size:12px;">Heatmap unavailable</span>'
                  }}
                />
              ) : (
                <span style={{ color: '#94a3b8', fontSize: '12px' }}>Heatmap not generated</span>
              )}
            </div>
          </div>

          {/* 3. Overlay */}
          <div
            onClick={() => overlayUrl && setActiveModalImage({ src: overlayUrl, alt: isMri ? 'Overlay (Grad-CAM on Brain MRI)' : 'Overlay (Grad-CAM on X-Ray)' })}
            style={{
              border: '1px solid var(--border)',
              borderRadius: '6px',
              padding: '12px',
              backgroundColor: '#f8fafc',
              cursor: overlayUrl ? 'pointer' : 'default',
              transition: 'border-color 0.15s ease'
            }}
          >
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--navy)', marginBottom: '8px' }}>
              3. Diagnostic Overlay
            </div>
            <div
              style={{
                width: '100%',
                height: '240px',
                backgroundColor: '#000000',
                borderRadius: '4px',
                overflow: 'hidden',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              {overlayUrl ? (
                <img
                  src={overlayUrl}
                  alt="Diagnostic Overlay"
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  onError={(e) => {
                    e.currentTarget.style.display = 'none'
                    e.currentTarget.parentElement.innerHTML = '<span style="color:#94a3b8;font-size:12px;">Overlay unavailable</span>'
                  }}
                />
              ) : (
                <span style={{ color: '#94a3b8', fontSize: '12px' }}>Overlay not generated</span>
              )}
            </div>
          </div>
        </div>

        {isMri && (
          <div
            style={{
              marginTop: '14px',
              padding: '10px 14px',
              backgroundColor: '#f8fafc',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              fontSize: '12.5px',
              color: 'var(--text-muted)',
              lineHeight: '1.5'
            }}
          >
            <strong style={{ color: 'var(--navy)' }}>Grad-CAM Limitation:</strong> Grad-CAM shows regions influencing the model; it is not verified tumour segmentation or clinical localisation. Intensity may spread across adjacent tissue.
          </div>
        )}
      </div>

      {/* Medical Disclaimer Banner */}
      <div
        style={{
          borderLeft: '4px solid var(--warning)',
          backgroundColor: '#fffbeb',
          padding: '16px 20px',
          borderRadius: '4px',
          marginBottom: '28px',
          fontSize: '13px',
          color: '#92400e',
          lineHeight: '1.5'
        }}
      >
        <strong>Research Disclaimer:</strong>{' '}
        {isMri
          ? 'This is an AI-generated research result and is not a confirmed medical diagnosis. It must not replace professional radiological interpretation.'
          : 'This is an AI-generated research result. Not a confirmed medical diagnosis. It must not replace professional radiological interpretation or licensed medical consultation.'}
      </div>

      {/* Lightbox Modal */}
      {activeModalImage && (
        <ImageModal
          src={activeModalImage.src}
          alt={activeModalImage.alt}
          onClose={() => setActiveModalImage(null)}
        />
      )}
    </div>
  )
}
