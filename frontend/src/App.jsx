import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import NewAnalysis from './pages/NewAnalysis'
import NewMriAnalysis from './pages/NewMriAnalysis'
import AnalysisDetails from './pages/AnalysisDetails'
import History from './pages/History'
import Settings from './pages/Settings'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/analysis/new" element={<NewAnalysis />} />
        <Route path="/mri/new" element={<NewMriAnalysis />} />
        <Route path="/analysis/:id" element={<AnalysisDetails />} />
        <Route path="/history" element={<History />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  )
}
