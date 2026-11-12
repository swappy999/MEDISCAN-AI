const BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000'

async function postAnalysisRequest(endpoint, formData, fallbackError) {
  const r = await fetch(`${BASE}${endpoint}`, { method: 'POST', body: formData })
  const data = await r.json()
  if (!r.ok) {
    const err = new Error(data.error || fallbackError)
    err.status = r.status
    err.validation = data.validation || null
    err.data = data
    throw err
  }
  return data
}

export const api = {
  health: () =>
    fetch(`${BASE}/api/health`).then((r) => {
      if (!r.ok) throw new Error(`Health check failed: ${r.statusText}`)
      return r.json()
    }),
  getAnalyses: (modality = null) => {
    const query = modality ? `?modality=${encodeURIComponent(modality)}` : ''
    return fetch(`${BASE}/api/analysis${query}`).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch analyses: ${r.statusText}`)
      return r.json()
    })
  },
  getAnalysis: (id) =>
    fetch(`${BASE}/api/analysis/${id}`).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch analysis: ${r.statusText}`)
      return r.json()
    }),
  deleteAnalysis: (id) =>
    fetch(`${BASE}/api/analysis/${id}`, { method: 'DELETE' }).then((r) => {
      if (!r.ok) throw new Error(`Failed to delete analysis: ${r.statusText}`)
      return r.json()
    }),
  reportUrl: (id) => `${BASE}/api/analysis/${id}/report`,
  resultUrl: (filename) => `${BASE}/api/results/${filename}`,
  validate: (formData) =>
    fetch(`${BASE}/api/validate`, { method: 'POST', body: formData }).then((r) => r.json()),
  validateMri: (formData) =>
    fetch(`${BASE}/api/mri/validate`, { method: 'POST', body: formData }).then((r) => r.json()),
  analyze: (formData) =>
    postAnalysisRequest('/api/analysis', formData, 'Chest X-ray analysis failed'),
  analyzeMri: (formData) =>
    postAnalysisRequest('/api/mri/analysis', formData, 'Brain MRI analysis failed'),
}
