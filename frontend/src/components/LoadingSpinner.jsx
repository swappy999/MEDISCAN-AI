import React from 'react'

export default function LoadingSpinner({ message = 'Loading...', size = 36 }) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '32px 16px',
        gap: '16px'
      }}
    >
      <div
        style={{
          width: `${size}px`,
          height: `${size}px`,
          border: '3px solid var(--border)',
          borderTopColor: 'var(--navy)',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite'
        }}
      />
      {message && (
        <p
          style={{
            color: 'var(--text-muted)',
            fontSize: '14px',
            fontWeight: 500,
            letterSpacing: '0.01em'
          }}
        >
          {message}
        </p>
      )}
    </div>
  )
}
