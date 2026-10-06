import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Upload, Search, FileText, Activity, AlertCircle } from 'lucide-react'
import { apiService } from '../services/api'

const HomePage = () => {
  const [systemInfo, setSystemInfo] = useState(null)
  const [stats, setStats] = useState(null)
  const [serverStatus, setServerStatus] = useState('checking')
  const [loading, setLoading] = useState(true)
  const [initMessage, setInitMessage] = useState(null)

  useEffect(() => {
    checkServerStatus()
  }, [])

  const checkServerStatus = async () => {
    try {
      setLoading(true)
      const [health, statsData] = await Promise.all([
        apiService.healthCheck(),
        apiService.getStats().catch(() => null)
      ])
      
      if (health.status === 'ok') {
        setServerStatus('online')
        setSystemInfo(health.data)
        if (statsData) {
          setStats(statsData)
        }
      } else {
        setServerStatus('offline')
      }
    } catch (error) {
      setServerStatus('offline')
      console.error('Server health check failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusIcon = () => {
    switch (serverStatus) {
      case 'online':
        return <Activity className="status-icon online" size={16} />
      case 'offline':
        return <AlertCircle className="status-icon offline" size={16} />
      default:
        return <div className="status-icon checking" />
    }
  }

  const getStatusText = () => {
    switch (serverStatus) {
      case 'online':
        return '服务正常'
      case 'offline':
        return '服务离线'
      default:
        return '检查中...'
    }
  }

  const handleInitialize = async () => {
    try {
      setLoading(true)
      setInitMessage(null)
      const result = await apiService.initializeIndex()
      setInitMessage({ type: 'success', text: result.message })
      // Refresh system info
      await checkServerStatus()
    } catch (error) {
      setInitMessage({ type: 'error', text: error.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-container">
      <div className="welcome-section">
        <h1 className="page-title">欢迎使用生物实验检索系统</h1>
        <p className="page-description">
          基于LlamaIndex构建的智能检索平台，帮助您快速查找生物实验相关信息
        </p>

        {/* Server Status */}
        <div className="status-card">
          <div className="status-header">
            {getStatusIcon()}
            <span className="status-text">{getStatusText()}</span>
            <button 
              className="refresh-btn"
              onClick={checkServerStatus}
              disabled={loading}
            >
              刷新状态
            </button>
          </div>

          {systemInfo && (
            <div className="system-info">
              <h3>系统信息</h3>
              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">系统名称:</span>
                  <span className="info-value">{systemInfo.message}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">版本:</span>
                  <span className="info-value">{systemInfo.version}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">描述:</span>
                  <span className="info-value">{systemInfo.description}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">索引状态:</span>
                  <span className="info-value">
                    {systemInfo.index_initialized ? '✅ 已初始化' : '❌ 未初始化'}
                  </span>
                </div>
                {stats && (
                  <div className="info-item">
                    <span className="info-label">向量总数:</span>
                    <span className="info-value">{stats.vector_count.toLocaleString()} 个</span>
                  </div>
                )}
              </div>
              
              {!systemInfo.index_initialized && (
                <div className="init-section">
                  <p className="init-message">索引未初始化，请先初始化索引以使用检索功能</p>
                  <button 
                    className="init-btn"
                    onClick={handleInitialize}
                    disabled={loading}
                  >
                    {loading ? '初始化中...' : '初始化索引'}
                  </button>
                  {initMessage && (
                    <div className={`init-result ${initMessage.type}`}>
                      {initMessage.text}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="quick-actions">
        <h2>快速操作</h2>
        <div className="actions-grid">
          <Link to="/upload" className="action-card">
            <div className="action-icon">
              <Upload size={32} />
            </div>
            <h3>上传文档</h3>
            <p>上传实验文献、论文或技术文档到系统中</p>
          </Link>

          <Link to="/search" className="action-card">
            <div className="action-icon">
              <Search size={32} />
            </div>
            <h3>智能检索</h3>
            <p>使用自然语言查询实验方法、步骤和材料信息</p>
          </Link>

          <Link to="/documents" className="action-card">
            <div className="action-icon">
              <FileText size={32} />
            </div>
            <h3>文档管理</h3>
            <p>查看和管理已上传的文档，浏览文档内容</p>
          </Link>
        </div>
      </div>

      {/* Features */}
      <div className="features-section">
        <h2>功能特性</h2>
        <div className="features-grid">
          <div className="feature-item">
            <h4>🔍 智能检索</h4>
            <p>支持自然语言查询，快速找到相关的实验信息</p>
          </div>
          <div className="feature-item">
            <h4>📄 多格式支持</h4>
            <p>支持PDF、Word、Markdown等多种文档格式</p>
          </div>
          <div className="feature-item">
            <h4>🧠 语义理解</h4>
            <p>基于LlamaIndex的语义检索，理解查询意图</p>
          </div>
          <div className="feature-item">
            <h4>⚡ 快速响应</h4>
            <p>优化的索引结构，提供毫秒级查询响应</p>
          </div>
        </div>
      </div>

      <style jsx>{`
        .welcome-section {
          text-align: center;
          margin-bottom: 3rem;
        }

        .status-card {
          background: white;
          border-radius: 12px;
          padding: 1.5rem;
          margin: 2rem auto;
          max-width: 600px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e2e8f0;
        }

        .status-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }

        .status-icon {
          border-radius: 50%;
        }

        .status-icon.online {
          color: #38a169;
        }

        .status-icon.offline {
          color: #e53e3e;
        }

        .status-icon.checking {
          width: 16px;
          height: 16px;
          background: #cbd5e0;
          border-radius: 50%;
          animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .status-text {
          font-weight: 500;
          flex: 1;
        }

        .refresh-btn {
          padding: 0.5rem 1rem;
          background: #f7fafc;
          border: 1px solid #e2e8f0;
          border-radius: 6px;
          cursor: pointer;
          font-size: 0.875rem;
          transition: all 0.2s;
        }

        .refresh-btn:hover {
          background: #edf2f7;
        }

        .refresh-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .system-info h3 {
          margin-bottom: 1rem;
          color: #2d3748;
        }

        .info-grid {
          display: grid;
          gap: 0.75rem;
        }

        .info-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem 0;
          border-bottom: 1px solid #f7fafc;
        }

        .info-label {
          font-weight: 500;
          color: #4a5568;
        }

        .info-value {
          color: #2d3748;
        }

        .init-section {
          margin-top: 1.5rem;
          padding-top: 1.5rem;
          border-top: 1px solid #e2e8f0;
          text-align: center;
        }

        .init-message {
          color: #718096;
          margin-bottom: 1rem;
        }

        .init-btn {
          background: #3182ce;
          color: white;
          border: none;
          border-radius: 8px;
          padding: 0.75rem 1.5rem;
          cursor: pointer;
          font-size: 1rem;
          font-weight: 500;
          transition: all 0.2s;
        }

        .init-btn:hover:not(:disabled) {
          background: #2c5aa0;
        }

        .init-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .init-result {
          margin-top: 1rem;
          padding: 0.75rem;
          border-radius: 6px;
          font-size: 0.9rem;
        }

        .init-result.success {
          background: #c6f6d5;
          color: #22543d;
          border: 1px solid #9ae6b4;
        }

        .init-result.error {
          background: #fed7d7;
          color: #742a2a;
          border: 1px solid #fc8181;
        }

        .quick-actions h2,
        .features-section h2 {
          margin-bottom: 1.5rem;
          color: #2d3748;
        }

        .actions-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 1.5rem;
          margin-bottom: 3rem;
        }

        .action-card {
          background: white;
          border-radius: 12px;
          padding: 2rem;
          text-align: center;
          text-decoration: none;
          color: inherit;
          transition: all 0.2s;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e2e8f0;
        }

        .action-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .action-icon {
          color: #3182ce;
          margin-bottom: 1rem;
        }

        .action-card h3 {
          margin-bottom: 0.5rem;
          color: #2d3748;
        }

        .action-card p {
          color: #718096;
          font-size: 0.9rem;
          line-height: 1.5;
        }

        .features-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.5rem;
        }

        .feature-item {
          background: white;
          border-radius: 8px;
          padding: 1.5rem;
          border: 1px solid #e2e8f0;
        }

        .feature-item h4 {
          margin-bottom: 0.5rem;
          color: #2d3748;
        }

        .feature-item p {
          color: #718096;
          font-size: 0.9rem;
          line-height: 1.5;
        }

        @media (max-width: 768px) {
          .actions-grid {
            grid-template-columns: 1fr;
          }
          
          .features-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  )
}

export default HomePage