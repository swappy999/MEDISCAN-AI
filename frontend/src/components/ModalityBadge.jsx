import React from 'react'

export default function ModalityBadge({ modality }) {
  const isMri = modality === 'brain_mri'
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '3px 9px',
        borderRadius: '4px',
        fontSize: '11.5px',
        fontWeight: 600,
        letterSpacing: '0.02em',
        backgroundColor: isMri ? '#f0f9ff' : '#f1f5f9',
        color: isMri ? '#0369a1' : 'var(--navy)',
        border: `1px solid ${isMri ? '#bae6fd' : '#cbd5e1'}`,
        whiteSpace: 'nowrap',
      }}
    >
      {isMri ? 'Brain MRI' : 'Chest X-Ray'}
    </span>
  )
}

