import React, { useState, useEffect } from 'react'
import { api } from '../services/api'
import LoadingSpinner from '../components/LoadingSpinner'

export default function Settings() {
  const [healthData, setHealthData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [checking, setChecking] = useState(false)
  const [lastChecked, setLastChecked] = useState(null)
  const [error, setError] = useState(null)

  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000'

  const runHealthCheck = async () => {
    try {
      setChecking(true)
      setError(null)
      const data = await api.health()
      setHealthData(data)
      setLastChecked(new Date())
    } catch (err) {
      setError(err.message || 'Health check failed: unable to connect to backend service')
      setHealthData(null)
      setLastChecked(new Date())
    } finally {
      setChecking(false)
      setLoading(false)
    }
  }

  useEffect(() => {
    runHealthCheck()
  }, [])

  const isBackendOk = healthData && healthData.status === 'ok'
  const mlService = healthData && healthData.mlService
  const mriService = healthData && healthData.mriService
  const isMlOk = mlService && (mlService.status === 'ok' || mlService.model_loaded === true)
  const isMriModelOk = mriService && mriService.model_loaded === true
  const isMriValidatorOk = mriService && mriService.validator_ready === true
  const isMriOk = Boolean(isMriModelOk && isMriValidatorOk)

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '28px' }}>
        <div>
          <h1 style={{ fontSize: '26px', color: 'var(--navy)', marginBottom: '4px' }}>Settings & System Status</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
            System configurations, service interconnects, and model diagnostics
          </p>
        </div>

        <button
          type="button"
          onClick={runHealthCheck}
          disabled={checking}
          style={{
            backgroundColor: 'var(--navy)',
            color: '#ffffff',
            border: 'none',
            padding: '9px 18px',
            borderRadius: '6px',
            fontSize: '13.5px',
            fontWeight: 600,
            cursor: checking ? 'not-allowed' : 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
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
            style={{ animation: checking ? 'spin 1s linear infinite' : 'none' }}
          >
            <polyline points="23 4 23 10 17 10"></polyline>
            <polyline points="1 20 1 14 7 14"></polyline>
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
          </svg>
          {checking ? 'Checking...' : 'Check Status'}
        </button>
      </div>

      {lastChecked && (
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '16px', textAlign: 'right' }}>
          Last checked: {lastChecked.toLocaleTimeString()}
        </div>
      )}

      {error && (
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
            gap: '10px'
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <LoadingSpinner message="Polling system diagnostics..." />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Service Configuration */}
          <div
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: '8px',
              padding: '24px',
              boxShadow: '0 1px 2px rgba(0,0,0,0.02)'
            }}
          >
            <h2 style={{ fontSize: '16px', color: 'var(--navy)', marginBottom: '16px' }}>Service Configuration</h2>

            <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: '12px 16px', fontSize: '13.5px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Backend API URL:</span>
              <span style={{ fontWeight: 600, fontFamily: 'monospace' }}>{apiUrl}</span>

              <span style={{ color: 'var(--text-muted)' }}>Chest X-Ray ML Host:</span>
              <span style={{ fontWeight: 600, fontFamily: 'monospace' }}>
                {mlService?.service_url || 'http://127.0.0.1:8000'}
              </span>

              <span style={{ color: 'var(--text-muted)' }}>Brain MRI ML Host:</span>
              <span style={{ fontWeight: 600, fontFamily: 'monospace' }}>
                {mriService?.service_url || 'http://127.0.0.1:8001'}
              </span>

              <span style={{ color: 'var(--text-muted)' }}>Environment Mode:</span>
              <span style={{ fontWeight: 500 }}>
                {import.meta.env.MODE || 'development'}
              </span>

              <span style={{ color: 'var(--text-muted)' }}>Authentication:</span>
              <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                Disabled (Local standalone medical software)
              </span>
            </div>
          </div>

          {/* Live Status Diagnostics */}
          <div
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: '8px',
              padding: '24px',
              boxShadow: '0 1px 2px rgba(0,0,0,0.02)'
            }}
          >
            <h2 style={{ fontSize: '16px', color: 'var(--navy)', marginBottom: '16px' }}>Live Interconnect Diagnostics</h2>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
              {/* Backend API Box */}
              <div
                style={{
                  padding: '18px',
                  borderRadius: '6px',
                  border: `1px solid ${isBackendOk ? '#bbf7d0' : '#fecaca'}`,
                  backgroundColor: isBackendOk ? '#f0fdf4' : '#fef2f2'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <div style={{ fontWeight: 600, fontSize: '14px', color: 'var(--navy)' }}>Backend Express API</div>
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '12px',
                      fontWeight: 600,
                      padding: '2px 8px',
                      borderRadius: '9999px',
                      backgroundColor: isBackendOk ? '#dcfce7' : '#fee2e2',
                      color: isBackendOk ? 'var(--success)' : 'var(--danger)'
                    }}
                  >
                    <span
                      style={{
                        width: '7px',
                        height: '7px',
                        borderRadius: '50%',
                        backgroundColor: isBackendOk ? 'var(--success)' : 'var(--danger)'
                      }}
                    />
                    {isBackendOk ? 'ONLINE' : 'OFFLINE'}
                  </span>
                </div>
                <div style={{ fontSize: '12.5px', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                  Port: 5000 &bull; Database: SQLite (mediscan.db) &bull; Multer uploads &bull; PDF generation
                </div>
              </div>

              {/* Chest X-Ray ML Service Box */}
              <div
                style={{
                  padding: '18px',
                  borderRadius: '6px',
                  border: `1px solid ${isMlOk ? '#bbf7d0' : '#fecaca'}`,
                  backgroundColor: isMlOk ? '#f0fdf4' : '#fef2f2'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <div style={{ fontWeight: 600, fontSize: '14px', color: 'var(--navy)' }}>Chest X-Ray FastAPI (:8000)</div>
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '12px',
                      fontWeight: 600,
                      padding: '2px 8px',
                      borderRadius: '9999px',
                      backgroundColor: isMlOk ? '#dcfce7' : '#fee2e2',
                      color: isMlOk ? 'var(--success)' : 'var(--danger)'
                    }}
                  >
                    <span
                      style={{
                        width: '7px',
                        height: '7px',
                        borderRadius: '50%',
                        backgroundColor: isMlOk ? 'var(--success)' : 'var(--danger)'
                      }}
                    />
                    {isMlOk ? 'MODEL LOADED' : 'UNAVAILABLE'}
                  </span>
                </div>
                <div style={{ fontSize: '12.5px', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                  Device: {mlService?.device || 'Unknown'} &bull; Architecture: {mlService?.model_name || 'DenseNet-121'} &bull; Validator: {(mlService?.validator_ready ?? mlService?.validator_loaded) ? 'Ready' : 'Not Verified'}
                  {mlService?.checkpoint_path && (
                    <div style={{ marginTop: '4px', wordBreak: 'break-all', fontFamily: 'monospace', fontSize: '11px' }}>
                      Checkpoint: {mlService.checkpoint_path}
                    </div>
                  )}
                  {mlService?.error && (
                    <div style={{ marginTop: '4px', color: 'var(--danger)', fontSize: '12px' }}>
                      Error: {mlService.error}
                    </div>
                  )}
                </div>
              </div>

              {/* Brain MRI ML Service Box */}
              <div
                style={{
                  padding: '18px',
                  borderRadius: '6px',
                  border: `1px solid ${isMriOk ? '#bbf7d0' : isMriModelOk ? '#fde68a' : '#fecaca'}`,
                  backgroundColor: isMriOk ? '#f0fdf4' : isMriModelOk ? '#fffbeb' : '#fef2f2'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <div style={{ fontWeight: 600, fontSize: '14px', color: 'var(--navy)' }}>Brain MRI FastAPI (:8001)</div>
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '12px',
                      fontWeight: 600,
                      padding: '2px 8px',
                      borderRadius: '9999px',
                      backgroundColor: isMriOk ? '#dcfce7' : isMriModelOk ? '#fef3c7' : '#fee2e2',
                      color: isMriOk ? 'var(--success)' : isMriModelOk ? 'var(--warning)' : 'var(--danger)'
                    }}
                  >
                    <span
                      style={{
                        width: '7px',
                        height: '7px',
                        borderRadius: '50%',
                        backgroundColor: isMriOk ? 'var(--success)' : isMriModelOk ? 'var(--warning)' : 'var(--danger)'
                      }}
                    />
                    {isMriOk ? 'READY' : isMriModelOk ? 'VALIDATOR PENDING' : 'UNAVAILABLE'}
                  </span>
                </div>
                <div style={{ fontSize: '12.5px', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                  Device: {mriService?.device || 'Unknown'} &bull; Architecture: {mriService?.model_name || 'ResNet-18'} &bull; Classes: {(mriService?.classes || ['glioma', 'meningioma', 'pituitary', 'no_tumor']).join(', ')}
                  <div style={{ marginTop: '4px' }}>
                    Validator: {mriService?.validator_status || (isMriValidatorOk ? 'Trained & calibrated' : 'Unavailable')}
                  </div>
                  {mriService?.checkpoint_path && (
                    <div style={{ marginTop: '4px', wordBreak: 'break-all', fontFamily: 'monospace', fontSize: '11px' }}>
                      Checkpoint: {mriService.checkpoint_path}
                    </div>
                  )}
                  {mriService?.error && (
                    <div style={{ marginTop: '4px', color: 'var(--danger)', fontSize: '12px' }}>
                      Error: {mriService.error}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {healthData && (
              <details style={{ marginTop: '20px', fontSize: '12.5px' }}>
                <summary style={{ cursor: 'pointer', color: 'var(--navy)', fontWeight: 600 }}>
                  View Raw Health JSON Response
                </summary>
                <pre
                  style={{
                    backgroundColor: '#f8fafc',
                    padding: '12px',
                    borderRadius: '6px',
                    border: '1px solid var(--border)',
                    marginTop: '8px',
                    overflowX: 'auto',
                    fontSize: '12px'
                  }}
                >
                  {JSON.stringify(healthData, null, 2)}
                </pre>
              </details>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
