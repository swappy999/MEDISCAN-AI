import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import PredictionBadge from '../components/PredictionBadge'
import ModalityBadge from '../components/ModalityBadge'
import LoadingSpinner from '../components/LoadingSpinner'

export default function History() {
  const [analyses, setAnalyses] = useState([])
  const [modalityFilter, setModalityFilter] = useState('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [deletingId, setDeletingId] = useState(null)

  const fetchAnalyses = async (filter = modalityFilter) => {
    try {
      setLoading(true)
      setError(null)
      const data = await api.getAnalyses(filter === 'all' ? undefined : filter)
      const list = Array.isArray(data) ? data : (data.analyses || data.data || [])
      setAnalyses(list)
    } catch (err) {
      setError(err.message || 'Failed to fetch analysis history')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAnalyses(modalityFilter)
  }, [modalityFilter])

  const handleDelete = async (id, filename) => {
    const confirmed = window.confirm(
      `Are you sure you want to delete analysis for "${filename || id}"? This will permanently remove the record and associated images.`
    )
    if (!confirmed) return

    try {
      setDeletingId(id)
      await api.deleteAnalysis(id)
      setAnalyses((prev) => prev.filter((a) => a.id !== id))
    } catch (err) {
      alert(`Delete failed: ${err.message}`)
    } finally {
      setDeletingId(null)
    }
  }

  const sortedAnalyses = [...analyses].sort(
    (a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0)
  )

  if (loading) {
    return <LoadingSpinner message="Loading historical analyses..." />
  }

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h1 style={{ fontSize: '26px', color: 'var(--navy)', marginBottom: '4px' }}>Analysis History</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
            Browse and manage all previous chest X-ray and brain MRI evaluations ({analyses.length} shown)
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <Link
            to="/analysis/new"
            style={{
              backgroundColor: 'var(--navy)',
              color: '#ffffff',
              padding: '10px 16px',
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
            Chest X-Ray
          </Link>
          <Link
            to="/mri/new"
            style={{
              backgroundColor: 'var(--cyan)',
              color: '#ffffff',
              padding: '10px 16px',
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
            Brain MRI
          </Link>
        </div>
      </div>

      {/* Modality Filter Pills */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '20px', flexWrap: 'wrap' }}>
        {[
          { key: 'all', label: 'All Modalities' },
          { key: 'chest_xray', label: 'Chest X-Ray' },
          { key: 'brain_mri', label: 'Brain MRI' }
        ].map((tab) => {
          const active = modalityFilter === tab.key
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => setModalityFilter(tab.key)}
              style={{
                padding: '7px 14px',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: 600,
                cursor: 'pointer',
                border: active ? '1px solid var(--navy)' : '1px solid var(--border)',
                backgroundColor: active ? 'var(--navy)' : 'var(--surface)',
                color: active ? '#ffffff' : 'var(--text-muted)'
              }}
            >
              {tab.label}
            </button>
          )
        })}
      </div>

      {error && (
        <div
          style={{
            padding: '12px 16px',
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            color: 'var(--danger)',
            borderRadius: '6px',
            marginBottom: '20px',
            fontSize: '13.5px'
          }}
        >
          {error}
        </div>
      )}

      {/* History Table or Empty State */}
      <div
        style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
          overflow: 'hidden'
        }}
      >
        {sortedAnalyses.length === 0 ? (
          <div style={{ padding: '64px 24px', textAlign: 'center' }}>
            <div
              style={{
                width: '52px',
                height: '52px',
                margin: '0 auto 16px auto',
                borderRadius: '50%',
                backgroundColor: '#f1f5f9',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-muted)'
              }}
            >
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
              </svg>
            </div>
            <h3 style={{ fontSize: '16px', color: 'var(--navy)', marginBottom: '6px' }}>No History Available</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '13.5px', marginBottom: '20px' }}>
              No analyses match this view yet. Run a chest X-ray or brain MRI analysis to begin tracking results.
            </p>
            <div style={{ display: 'inline-flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'center' }}>
              <Link
                to="/analysis/new"
                style={{
                  backgroundColor: 'var(--navy)',
                  color: '#ffffff',
                  padding: '9px 18px',
                  borderRadius: '6px',
                  fontSize: '13.5px',
                  fontWeight: 500,
                  display: 'inline-block'
                }}
              >
                New Chest X-Ray
              </Link>
              <Link
                to="/mri/new"
                style={{
                  backgroundColor: 'var(--cyan)',
                  color: '#ffffff',
                  padding: '9px 18px',
                  borderRadius: '6px',
                  fontSize: '13.5px',
                  fontWeight: 500,
                  display: 'inline-block'
                }}
              >
                New Brain MRI
              </Link>
            </div>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13.5px' }}>
              <thead>
                <tr style={{ backgroundColor: 'var(--navy)', color: '#ffffff', fontSize: '12.5px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  <th style={{ padding: '12px 18px', fontWeight: 600 }}>Timestamp</th>
                  <th style={{ padding: '12px 18px', fontWeight: 600 }}>Modality</th>
                  <th style={{ padding: '12px 18px', fontWeight: 600 }}>Filename</th>
                  <th style={{ padding: '12px 18px', fontWeight: 600 }}>Prediction</th>
                  <th style={{ padding: '12px 18px', fontWeight: 600 }}>Confidence</th>
                  <th style={{ padding: '12px 18px', fontWeight: 600, textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {sortedAnalyses.map((item, index) => {
                  const conf = item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : 'N/A'
                  const dateStr = item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A'
                  const isEven = index % 2 === 0
                  const isDeleting = deletingId === item.id

                  return (
                    <tr
                      key={item.id}
                      style={{
                        backgroundColor: isEven ? '#ffffff' : '#f8fafc',
                        borderBottom: '1px solid var(--border)',
                        opacity: isDeleting ? 0.5 : 1
                      }}
                    >
                      <td style={{ padding: '14px 18px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                        {dateStr}
                      </td>
                      <td style={{ padding: '14px 18px' }}>
                        <ModalityBadge modality={item.modality} />
                      </td>
                      <td
                        style={{
                          padding: '14px 18px',
                          fontWeight: 500,
                          maxWidth: '220px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}
                        title={item.original_filename}
                      >
                        {item.original_filename || 'image.jpg'}
                      </td>
                      <td style={{ padding: '14px 18px' }}>
                        <PredictionBadge prediction={item.prediction} />
                      </td>
                      <td style={{ padding: '14px 18px', fontWeight: 600, color: 'var(--navy)' }}>
                        {conf}
                      </td>
                      <td style={{ padding: '14px 18px', textAlign: 'right' }}>
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                          <Link
                            to={`/analysis/${item.id}`}
                            style={{
                              padding: '5px 10px',
                              backgroundColor: '#f1f5f9',
                              color: 'var(--navy)',
                              borderRadius: '4px',
                              fontSize: '12.5px',
                              fontWeight: 600
                            }}
                          >
                            View
                          </Link>

                          <a
                            href={api.reportUrl(item.id)}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{
                              padding: '5px 10px',
                              backgroundColor: '#f1f5f9',
                              color: 'var(--text)',
                              borderRadius: '4px',
                              fontSize: '12.5px',
                              fontWeight: 500
                            }}
                          >
                            PDF
                          </a>

                          <button
                            type="button"
                            onClick={() => handleDelete(item.id, item.original_filename)}
                            disabled={isDeleting}
                            style={{
                              padding: '5px 10px',
                              backgroundColor: '#fef2f2',
                              border: '1px solid #fecaca',
                              color: 'var(--danger)',
                              borderRadius: '4px',
                              fontSize: '12.5px',
                              fontWeight: 500,
                              cursor: isDeleting ? 'not-allowed' : 'pointer'
                            }}
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
