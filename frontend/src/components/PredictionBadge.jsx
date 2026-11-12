import React from 'react'

function resolveBadgeStyle(prediction) {
  const norm = String(prediction || '').toUpperCase()
  if (norm === 'NORMAL' || norm === 'NO_TUMOR' || norm === 'NOTUMOR') {
    return {
      label: norm === 'NORMAL' ? 'NORMAL' : 'NO TUMOR',
      bg: '#f0fdf4',
      color: 'var(--success)',
      border: '#bbf7d0',
    }
  }
  if (norm === 'MENINGIOMA') {
    return {
      label: 'MENINGIOMA',
      bg: '#fffbeb',
      color: 'var(--warning)',
      border: '#fde68a',
    }
  }
  if (norm === 'PNEUMONIA' || norm === 'GLIOMA' || norm === 'PITUITARY') {
    return {
      label: norm,
      bg: '#fef2f2',
      color: 'var(--danger)',
      border: '#fecaca',
    }
  }
  return {
    label: prediction || 'UNKNOWN',
    bg: '#f8fafc',
    color: 'var(--text-muted)',
    border: 'var(--border)',
  }
}

export default function PredictionBadge({ prediction, size = 'normal' }) {
  const isLarge = size === 'large'
  const meta = resolveBadgeStyle(prediction)

  const style = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: isLarge ? '8px' : '6px',
    padding: isLarge ? '6px 16px' : '3px 10px',
    borderRadius: '9999px',
    fontSize: isLarge ? '15px' : '12px',
    fontWeight: '600',
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
    backgroundColor: meta.bg,
    color: meta.color,
    border: `1px solid ${meta.border}`,
    whiteSpace: 'nowrap',
  }

  const dotStyle = {
    width: isLarge ? '10px' : '7px',
    height: isLarge ? '10px' : '7px',
    borderRadius: '50%',
    backgroundColor: meta.color,
  }

  return (
    <span style={style}>
      <span style={dotStyle} />
      {meta.label}
    </span>
  )
}
