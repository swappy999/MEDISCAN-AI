import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import PredictionBadge from '../components/PredictionBadge'
import ModalityBadge from '../components/ModalityBadge'
import LoadingSpinner from '../components/LoadingSpinner'

export default function Dashboard() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [analyses, setAnalyses] = useState([])
  const [health, setHealth] = useState({ backend: false, cxrService: false, mriService: false, details: null })

  useEffect(() => {
    let isMounted = true

    async function loadDashboardData() {
      try {
        setLoading(true)
        setError(null)

        // Run health check and fetch analyses in parallel
        const [healthRes, analysesRes] = await Promise.allSettled([
          api.health(),
          api.getAnalyses()
        ])

        if (isMounted) {
          if (healthRes.status === 'fulfilled' && healthRes.value) {
            const h = healthRes.value
            const cxrOk = h.mlService && (h.mlService.status === 'ok' || h.mlService.model_loaded === true)
            const mriOk = h.mriService && (h.mriService.status === 'ok' || h.mriService.model_loaded === true)
            setHealth({
              backend: h.status === 'ok',
              cxrService: Boolean(cxrOk),
              mriService: Boolean(mriOk),
              details: h
            })
          } else {
            setHealth({ backend: false, cxrService: false, mriService: false, details: null })
          }

          if (analysesRes.status === 'fulfilled') {
            const data = analysesRes.value
            const list = Array.isArray(data) ? data : (data.analyses || data.data || [])
            setAnalyses(list)
          } else {
            setAnalyses([])
          }
        }
      } catch (err) {
        if (isMounted) setError(err.message || 'Failed to load dashboard data')
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    loadDashboardData()
    return () => { isMounted = false }
  }, [])

  const totalAnalyses = analyses.length
  const normalCount = analyses.filter((a) => {
    const p = String(a.prediction || '').toUpperCase()
    return p === 'NORMAL' || p === 'NO_TUMOR' || p === 'NOTUMOR'
  }).length
  const findingCount = analyses.filter((a) => {
    const p = String(a.prediction || '').toUpperCase()
    return p === 'PNEUMONIA' || p === 'GLIOMA' || p === 'MENINGIOMA' || p === 'PITUITARY'
  }).length

  const sortedAnalyses = [...analyses].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
  const recentAnalyses = sortedAnalyses.slice(0, 5)

  if (loading) {
    return <LoadingSpinner message="Loading dashboard overview..." />
  }

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '28px' }}>
        <div>
          <h1 style={{ fontSize: '26px', color: 'var(--navy)', marginBottom: '4px' }}>Dashboard</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
            System health status and radiological analysis metrics
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
              gap: '8px',
              boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
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
              gap: '8px',
              boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
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

      {/* Service Status Bar */}
      <div
        style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          padding: '14px 20px',
          display: 'flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '24px',
          marginBottom: '24px',
          boxShadow: '0 1px 2px rgba(0,0,0,0.02)'
        }}
      >
        <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Services:
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13.5px' }}>
          <span
            style={{
              width: '9px',
              height: '9px',
              borderRadius: '50%',
              backgroundColor: health.backend ? 'var(--success)' : 'var(--danger)'
            }}
          />
          <span style={{ fontWeight: 500 }}>Backend API (:5000)</span>
          <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            ({health.backend ? 'Online' : 'Offline'})
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13.5px' }}>
          <span
            style={{
              width: '9px',
              height: '9px',
              borderRadius: '50%',
              backgroundColor: health.cxrService ? 'var(--success)' : 'var(--danger)'
            }}
          />
          <span style={{ fontWeight: 500 }}>Chest X-Ray ML (:8000)</span>
          <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            ({health.cxrService ? 'Model Loaded' : 'Unavailable'})
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13.5px' }}>
          <span
            style={{
              width: '9px',
              height: '9px',
              borderRadius: '50%',
              backgroundColor:
                health.details?.mriService?.model_loaded && health.details?.mriService?.validator_ready
                  ? 'var(--success)'
                  : health.mriService
                    ? 'var(--warning)'
                    : 'var(--danger)'
            }}
          />
          <span style={{ fontWeight: 500 }}>Brain MRI ML (:8001)</span>
          <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            (
            {health.details?.mriService?.model_loaded && health.details?.mriService?.validator_ready
              ? 'Ready'
              : health.mriService
                ? 'Validator Not Ready'
                : 'Unavailable'}
            )
          </span>
        </div>
      </div>

      {/* Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '28px' }}>
        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            padding: '20px',
            boxShadow: '0 1px 2px rgba(0,0,0,0.02)'
          }}
        >
          <div style={{ color: 'var(--text-muted)', fontSize: '13px', fontWeight: 500 }}>Total Analyses</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: 'var(--navy)', marginTop: '6px' }}>
            {totalAnalyses}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>Recorded CXR &amp; MRI scans</div>
        </div>

        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            padding: '20px',
            boxShadow: '0 1px 2px rgba(0,0,0,0.02)'
          }}
        >
          <div style={{ color: 'var(--text-muted)', fontSize: '13px', fontWeight: 500 }}>Normal / No Tumor</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: 'var(--success)', marginTop: '6px' }}>
            {normalCount}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {totalAnalyses > 0 ? `${Math.round((normalCount / totalAnalyses) * 100)}% of total` : 'No cases'}
          </div>
        </div>

        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            padding: '20px',
            boxShadow: '0 1px 2px rgba(0,0,0,0.02)'
          }}
        >
          <div style={{ color: 'var(--text-muted)', fontSize: '13px', fontWeight: 500 }}>Findings Detected</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: 'var(--danger)', marginTop: '6px' }}>
            {findingCount}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {totalAnalyses > 0 ? `${Math.round((findingCount / totalAnalyses) * 100)}% of total` : 'No cases'}
          </div>
        </div>
      </div>

      {/* Recent Analyses Section */}
      <div
        style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
          overflow: 'hidden'
        }}
      >
        <div
          style={{
            padding: '18px 24px',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <div>
            <h2 style={{ fontSize: '16px', color: 'var(--navy)' }}>Recent Analyses</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '12.5px', marginTop: '2px' }}>
              Showing the latest radiological evaluation results
            </p>
          </div>
          {analyses.length > 5 && (
            <Link
              to="/history"
              style={{
                color: 'var(--cyan)',
                fontSize: '13px',
                fontWeight: 600
              }}
            >
              View all ({analyses.length}) &rarr;
            </Link>
          )}
        </div>

        {recentAnalyses.length === 0 ? (
          <div style={{ padding: '48px 24px', textAlign: 'center' }}>
            <div
              style={{
                width: '48px',
                height: '48px',
                margin: '0 auto 16px auto',
                borderRadius: '50%',
                backgroundColor: '#f1f5f9',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-muted)'
              }}
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
            </div>
            <p style={{ color: 'var(--navy)', fontWeight: 600, fontSize: '15px' }}>No analyses recorded yet</p>
            <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginTop: '4px', marginBottom: '20px' }}>
              Upload a chest X-ray or brain MRI scan to perform validated AI analysis with Grad-CAM visualization.
            </p>
            <div style={{ display: 'inline-flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'center' }}>
              <Link
                to="/analysis/new"
                style={{
                  backgroundColor: 'var(--navy)',
                  color: '#ffffff',
                  padding: '9px 16px',
                  borderRadius: '6px',
                  fontSize: '13.5px',
                  fontWeight: 500,
                  display: 'inline-block'
                }}
              >
                Analyze Chest X-Ray
              </Link>
              <Link
                to="/mri/new"
                style={{
                  backgroundColor: 'var(--cyan)',
                  color: '#ffffff',
                  padding: '9px 16px',
                  borderRadius: '6px',
                  fontSize: '13.5px',
                  fontWeight: 500,
                  display: 'inline-block'
                }}
              >
                Analyze Brain MRI
              </Link>
            </div>
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13.5px' }}>
            <thead>
              <tr style={{ backgroundColor: 'var(--navy)', color: '#ffffff', fontSize: '12.5px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                <th style={{ padding: '12px 20px', fontWeight: 600 }}>Timestamp</th>
                <th style={{ padding: '12px 20px', fontWeight: 600 }}>Modality</th>
                <th style={{ padding: '12px 20px', fontWeight: 600 }}>File</th>
                <th style={{ padding: '12px 20px', fontWeight: 600 }}>Prediction</th>
                <th style={{ padding: '12px 20px', fontWeight: 600 }}>Confidence</th>
                <th style={{ padding: '12px 20px', fontWeight: 600, textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {recentAnalyses.map((item, index) => {
                const confPercent = item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : 'N/A'
                const formattedDate = item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A'
                const isEven = index % 2 === 0
                return (
                  <tr
                    key={item.id}
                    style={{
                      backgroundColor: isEven ? '#ffffff' : '#f8fafc',
                      borderBottom: '1px solid var(--border)'
                    }}
                  >
                    <td style={{ padding: '14px 20px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                      {formattedDate}
                    </td>
                    <td style={{ padding: '14px 20px' }}>
                      <ModalityBadge modality={item.modality} />
                    </td>
                    <td style={{ padding: '14px 20px', fontWeight: 500, maxWidth: '240px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {item.original_filename || 'image.jpg'}
                    </td>
                    <td style={{ padding: '14px 20px' }}>
                      <PredictionBadge prediction={item.prediction} />
                    </td>
                    <td style={{ padding: '14px 20px', fontWeight: 600, color: 'var(--navy)' }}>
                      {confPercent}
                    </td>
                    <td style={{ padding: '14px 20px', textAlign: 'right' }}>
                      <Link
                        to={`/analysis/${item.id}`}
                        style={{
                          color: 'var(--cyan)',
                          fontWeight: 600,
                          fontSize: '13px'
                        }}
                      >
                        View Details &rarr;
                      </Link>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
