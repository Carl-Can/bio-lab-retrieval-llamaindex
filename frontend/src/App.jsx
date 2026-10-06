import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import { Upload, Search, FileText, Home } from 'lucide-react'
import HomePage from './components/HomePage'
import UploadPage from './components/UploadPage'
import SearchPage from './components/SearchPage'
import DocumentsPage from './components/DocumentsPage'
import { ErrorBoundary } from './components/ErrorHandling'
import './styles/App.css'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <Router>
      <div className="app">
        {/* Header */}
        <header className="header">
          <div className="header-content">
            <button 
              className="sidebar-toggle"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              ☰
            </button>
            <h1 className="app-title">生物实验检索系统</h1>
            <div className="header-subtitle">基于LlamaIndex构建的生物实验检索平台</div>
          </div>
        </header>

        <div className="main-container">
          {/* Sidebar */}
          <nav className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
            <div className="nav-menu">
              <NavLink to="/" className="nav-item" onClick={() => setSidebarOpen(false)}>
                <Home size={20} />
                <span>首页</span>
              </NavLink>
              <NavLink to="/upload" className="nav-item" onClick={() => setSidebarOpen(false)}>
                <Upload size={20} />
                <span>文档上传</span>
              </NavLink>
              <NavLink to="/search" className="nav-item" onClick={() => setSidebarOpen(false)}>
                <Search size={20} />
                <span>智能检索</span>
              </NavLink>
              <NavLink to="/documents" className="nav-item" onClick={() => setSidebarOpen(false)}>
                <FileText size={20} />
                <span>文档管理</span>
              </NavLink>
            </div>
          </nav>

          {/* Main Content */}
          <main className="content">
            <ErrorBoundary>
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/upload" element={<UploadPage />} />
                <Route path="/search" element={<SearchPage />} />
                <Route path="/documents" element={<DocumentsPage />} />
              </Routes>
            </ErrorBoundary>
          </main>
        </div>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div 
            className="sidebar-overlay"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </div>
    </Router>
  )
}

export default App