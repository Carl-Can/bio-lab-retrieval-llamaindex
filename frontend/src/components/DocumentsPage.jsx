import React, { useState, useEffect } from 'react'
import { FileText, Search, Eye, Download, Trash2, Calendar, Hash } from 'lucide-react'
import { apiService } from '../services/api'
import { formatFileSize } from '../utils/helpers'

const DocumentsPage = () => {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedDoc, setSelectedDoc] = useState(null)
  const [viewingChunks, setViewingChunks] = useState(false)

  useEffect(() => {
    loadDocuments()
  }, [])

  const loadDocuments = async () => {
    try {
      setLoading(true)
      const data = await apiService.getDocuments()
      setDocuments(data.documents || [])
    } catch (err) {
      console.error('Failed to load documents:', err)
      // Fallback to localStorage
      const uploadHistory = localStorage.getItem('uploadHistory')
      if (uploadHistory) {
        setDocuments(JSON.parse(uploadHistory))
      }
    } finally {
      setLoading(false)
    }
  }

  const handleViewDocument = async (doc) => {
    try {
      setLoading(true)
      const docData = await apiService.getDocument(doc.id)
      setSelectedDoc({ ...docData, filename: doc.filename })
      setViewingChunks(true)
    } catch (err) {
      setError(`获取文档失败: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteDocument = (docId) => {
    if (window.confirm('确定要删除这个文档吗？')) {
      // Remove from localStorage (in a real app, this would call the API)
      const updatedDocs = documents.filter(doc => doc.id !== docId)
      setDocuments(updatedDocs)
      localStorage.setItem('uploadHistory', JSON.stringify(updatedDocs))
    }
  }

  const filteredDocuments = documents.filter(doc =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (doc.description && doc.description.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  const getFileIcon = (filename) => {
    const ext = filename.split('.').pop()?.toLowerCase()
    switch (ext) {
      case 'pdf': return '📄'
      case 'doc':
      case 'docx': return '📝'
      case 'md': return '📋'
      case 'txt': return '📄'
      default: return '📄'
    }
  }

  if (viewingChunks && selectedDoc) {
    return (
      <div className="page-container">
        <div className="document-viewer">
          <div className="viewer-header">
            <button 
              className="back-btn"
              onClick={() => {
                setViewingChunks(false)
                setSelectedDoc(null)
              }}
            >
              ← 返回文档列表
            </button>
            <h2>文档内容详情</h2>
          </div>

          <div className="document-info">
            <h3>{selectedDoc.filename || selectedDoc.doc_id}</h3>
            <p>共 {selectedDoc.chunks?.length || 0} 个文档块</p>
          </div>

          <div className="chunks-list">
            {selectedDoc.chunks?.map((chunk, index) => (
              <div key={index} className="chunk-item">
                <div className="chunk-header">
                  <Hash size={16} />
                  <span>块 {chunk.chunk_id !== undefined ? chunk.chunk_id + 1 : index + 1}</span>
                </div>
                <div className="chunk-content">
                  {typeof chunk === 'string' ? chunk : chunk.text}
                </div>
              </div>
            )) || (
              <div className="no-chunks">
                <p>暂无文档块数据</p>
              </div>
            )}
          </div>
        </div>

        <style jsx>{`
          .document-viewer {
            max-width: 1000px;
            margin: 0 auto;
          }

          .viewer-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #e2e8f0;
          }

          .back-btn {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            cursor: pointer;
            color: #4a5568;
            transition: all 0.2s;
          }

          .back-btn:hover {
            background: #edf2f7;
            color: #2d3748;
          }

          .viewer-header h2 {
            margin: 0;
            color: #2d3748;
          }

          .document-info {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid #e2e8f0;
          }

          .document-info h3 {
            margin: 0 0 0.5rem 0;
            color: #2d3748;
          }

          .document-info p {
            margin: 0;
            color: #718096;
          }

          .chunks-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
          }

          .chunk-item {
            background: white;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            overflow: hidden;
          }

          .chunk-header {
            background: #f7fafc;
            padding: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
            font-weight: 500;
            color: #2d3748;
          }

          .chunk-content {
            padding: 1.5rem;
            line-height: 1.6;
            color: #4a5568;
            white-space: pre-wrap;
          }

          .no-chunks {
            text-align: center;
            padding: 3rem;
            color: #718096;
          }
        `}</style>
      </div>
    )
  }

  return (
    <div className="page-container">
      <h1 className="page-title">文档管理</h1>
      <p className="page-description">
        查看和管理已上传的文档，浏览文档内容
      </p>

      {/* Search Bar */}
      <div className="search-bar">
        <div className="search-input-wrapper">
          <Search size={20} className="search-icon" />
          <input
            type="text"
            placeholder="搜索文档名称..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
        </div>
      </div>

      {/* Documents List */}
      {loading ? (
        <div className="loading-container">
          <p>加载中...</p>
        </div>
      ) : error ? (
        <div className="error-container">
          <p>{error}</p>
        </div>
      ) : filteredDocuments.length === 0 ? (
        <div className="empty-state">
          <FileText size={64} className="empty-icon" />
          <h3>暂无文档</h3>
          <p>
            {searchQuery ? '没有找到匹配的文档' : '还没有上传任何文档'}
          </p>
          {!searchQuery && (
            <a href="/upload" className="btn btn-primary">
              上传文档
            </a>
          )}
        </div>
      ) : (
        <div className="documents-grid">
          {filteredDocuments.map((doc) => (
            <div key={doc.id} className="document-card">
              <div className="doc-header">
                <div className="doc-icon">
                  {getFileIcon(doc.filename)}
                </div>
                <div className="doc-info">
                  <h3 className="doc-title">{doc.filename}</h3>
                  <div className="doc-meta">
                    <span className="doc-size">
                      {formatFileSize(doc.size || 0)}
                    </span>
                    {doc.chunks_count && (
                      <>
                        <span className="separator">·</span>
                        <span className="doc-chunks">
                          {doc.chunks_count} 块
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {doc.description && (
                <div className="doc-description">
                  {doc.description}
                </div>
              )}

              <div className="doc-timestamp">
                <Calendar size={14} />
                <span>
                  {doc.upload_time 
                    ? new Date(doc.upload_time).toLocaleString('zh-CN')
                    : '上传时间未知'
                  }
                </span>
                <span className="separator">·</span>
                <span className={`doc-source ${doc.source}`}>
                  {doc.source === 'indexed' ? '索引库' : '已上传'}
                </span>
              </div>

              <div className="doc-actions">
                <button 
                  className="action-btn view-btn"
                  onClick={() => handleViewDocument(doc)}
                  title="查看文档内容"
                >
                  <Eye size={16} />
                  查看
                </button>
                
                <button 
                  className="action-btn download-btn"
                  onClick={() => {
                    // In a real app, this would download the file
                    alert('下载功能待实现')
                  }}
                  title="下载文档"
                >
                  <Download size={16} />
                  下载
                </button>
                
                <button 
                  className="action-btn delete-btn"
                  onClick={() => handleDeleteDocument(doc.id)}
                  title="删除文档"
                >
                  <Trash2 size={16} />
                  删除
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <style jsx>{`
        .search-bar {
          max-width: 600px;
          margin: 0 auto 2rem;
        }

        .search-input-wrapper {
          position: relative;
          display: flex;
          align-items: center;
        }

        .search-icon {
          position: absolute;
          left: 1rem;
          color: #718096;
          z-index: 1;
        }

        .search-input {
          width: 100%;
          padding: 0.75rem 1rem 0.75rem 3rem;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          font-size: 1rem;
          background: white;
          transition: border-color 0.2s;
        }

        .search-input:focus {
          outline: none;
          border-color: #3182ce;
          box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
        }

        .loading-container,
        .error-container {
          text-align: center;
          padding: 3rem;
          color: #718096;
        }

        .error-container {
          color: #c53030;
          background: #fed7d7;
          border-radius: 8px;
        }

        .empty-state {
          text-align: center;
          padding: 4rem 2rem;
          color: #718096;
        }

        .empty-icon {
          color: #cbd5e0;
          margin-bottom: 1rem;
        }

        .empty-state h3 {
          margin: 0 0 0.5rem 0;
          color: #4a5568;
        }

        .empty-state p {
          margin: 0 0 2rem 0;
        }

        .documents-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
          gap: 1.5rem;
        }

        .document-card {
          background: white;
          border-radius: 12px;
          padding: 1.5rem;
          border: 1px solid #e2e8f0;
          transition: all 0.2s;
          box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .document-card:hover {
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
          transform: translateY(-2px);
        }

        .doc-header {
          display: flex;
          align-items: flex-start;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .doc-icon {
          font-size: 2rem;
          flex-shrink: 0;
        }

        .doc-info {
          flex: 1;
          min-width: 0;
        }

        .doc-title {
          margin: 0 0 0.5rem 0;
          font-size: 1.1rem;
          font-weight: 600;
          color: #2d3748;
          word-break: break-word;
        }

        .doc-meta {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.875rem;
          color: #718096;
        }

        .separator {
          color: #cbd5e0;
        }

        .doc-description {
          margin-bottom: 1rem;
          padding: 0.75rem;
          background: #f7fafc;
          border-radius: 6px;
          font-size: 0.9rem;
          color: #4a5568;
          line-height: 1.5;
        }

        .doc-timestamp {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 1rem;
          font-size: 0.875rem;
          color: #718096;
        }

        .doc-source {
          padding: 0.125rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
          font-weight: 500;
        }

        .doc-source.indexed {
          background: #ebf8ff;
          color: #3182ce;
        }

        .doc-source.uploaded {
          background: #f0fff4;
          color: #38a169;
        }

        .doc-actions {
          display: flex;
          gap: 0.5rem;
          flex-wrap: wrap;
        }

        .action-btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          border: 1px solid #e2e8f0;
          border-radius: 6px;
          background: white;
          color: #4a5568;
          cursor: pointer;
          font-size: 0.875rem;
          transition: all 0.2s;
          text-decoration: none;
        }

        .action-btn:hover {
          border-color: #cbd5e0;
          background: #f7fafc;
        }

        .view-btn:hover {
          border-color: #3182ce;
          color: #3182ce;
          background: #ebf8ff;
        }

        .download-btn:hover {
          border-color: #38a169;
          color: #38a169;
          background: #f0fff4;
        }

        .delete-btn:hover {
          border-color: #e53e3e;
          color: #e53e3e;
          background: #fed7d7;
        }

        @media (max-width: 768px) {
          .documents-grid {
            grid-template-columns: 1fr;
          }

          .doc-actions {
            justify-content: center;
          }

          .action-btn {
            flex: 1;
            justify-content: center;
          }
        }
      `}</style>
    </div>
  )
}

export default DocumentsPage