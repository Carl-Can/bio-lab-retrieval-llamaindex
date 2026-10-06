import React, { useState, useRef } from 'react'
import { Upload, File, X, CheckCircle, AlertCircle, Loader } from 'lucide-react'
import { apiService } from '../services/api'
import { formatFileSize } from '../utils/helpers'

const UploadPage = () => {
  const [isDragOver, setIsDragOver] = useState(false)
  const [files, setFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const droppedFiles = Array.from(e.dataTransfer.files)
    addFiles(droppedFiles)
  }

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files)
    addFiles(selectedFiles)
  }

  const addFiles = (newFiles) => {
    const validFiles = newFiles.filter(file => {
      const isValidType = file.type === 'application/pdf' || 
                         file.type === 'application/msword' ||
                         file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
                         file.type === 'text/markdown' ||
                         file.type === 'text/plain'
      
      const isValidSize = file.size <= 10 * 1024 * 1024 // 10MB
      
      return isValidType && isValidSize
    })

    const filesWithStatus = validFiles.map(file => ({
      file,
      id: Date.now() + Math.random(),
      status: 'pending', // pending, uploading, success, error
      progress: 0,
      error: null,
      result: null
    }))

    setFiles(prev => [...prev, ...filesWithStatus])
  }

  const removeFile = (fileId) => {
    setFiles(prev => prev.filter(f => f.id !== fileId))
  }

  const uploadFile = async (fileItem) => {
    try {
      setFiles(prev => prev.map(f => 
        f.id === fileItem.id 
          ? { ...f, status: 'uploading', progress: 0 }
          : f
      ))

      const result = await apiService.uploadDocument(
        fileItem.file,
        (progress) => {
          setFiles(prev => prev.map(f => 
            f.id === fileItem.id 
              ? { ...f, progress }
              : f
          ))
        }
      )

      setFiles(prev => prev.map(f => 
        f.id === fileItem.id 
          ? { ...f, status: 'success', progress: 100, result }
          : f
      ))

      return result
    } catch (error) {
      setFiles(prev => prev.map(f => 
        f.id === fileItem.id 
          ? { ...f, status: 'error', error: error.message }
          : f
      ))
      throw error
    }
  }

  const uploadAllFiles = async () => {
    const pendingFiles = files.filter(f => f.status === 'pending')
    if (pendingFiles.length === 0) return

    setUploading(true)
    
    try {
      for (const fileItem of pendingFiles) {
        await uploadFile(fileItem)
      }
    } catch (error) {
      console.error('Upload failed:', error)
    } finally {
      setUploading(false)
    }
  }

  const clearAllFiles = () => {
    setFiles([])
  }

  const getFileIcon = (file) => {
    if (file.type === 'application/pdf') return '📄'
    if (file.type.includes('word')) return '📝'
    if (file.type === 'text/markdown') return '📋'
    return '📄'
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'uploading':
        return <Loader className="animate-spin" size={16} />
      case 'success':
        return <CheckCircle className="text-green-500" size={16} />
      case 'error':
        return <AlertCircle className="text-red-500" size={16} />
      default:
        return null
    }
  }

  const pendingCount = files.filter(f => f.status === 'pending').length
  const successCount = files.filter(f => f.status === 'success').length
  const errorCount = files.filter(f => f.status === 'error').length

  return (
    <div className="page-container">
      <h1 className="page-title">文档上传</h1>
      <p className="page-description">
        上传实验文献、论文或技术文档，系统将自动解析并建立索引
      </p>

      {/* Upload Area */}
      <div className="upload-section">
        <div 
          className={`upload-area ${isDragOver ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.doc,.docx,.md,.txt"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          
          <div className="upload-content">
            <Upload size={48} className="upload-icon" />
            <h3>拖拽文件到此处或点击选择文件</h3>
            <p>支持 PDF, Word, Markdown 格式，单个文件不超过 10MB</p>
          </div>
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="file-list">
            <div className="file-list-header">
              <h3>待上传文件 ({files.length})</h3>
              <div className="file-actions">
                {pendingCount > 0 && (
                  <button 
                    className="btn btn-primary"
                    onClick={uploadAllFiles}
                    disabled={uploading}
                  >
                    {uploading ? '上传中...' : `上传全部 (${pendingCount})`}
                  </button>
                )}
                <button className="btn btn-secondary" onClick={clearAllFiles}>
                  清空列表
                </button>
              </div>
            </div>

            <div className="files">
              {files.map((fileItem) => (
                <div key={fileItem.id} className="file-item">
                  <div className="file-info">
                    <span className="file-icon">
                      {getFileIcon(fileItem.file)}
                    </span>
                    <div className="file-details">
                      <div className="file-name">{fileItem.file.name}</div>
                      <div className="file-size">
                        {formatFileSize(fileItem.file.size)}
                      </div>
                    </div>
                  </div>

                  <div className="file-status">
                    {fileItem.status === 'uploading' && (
                      <div className="progress-container">
                        <div className="progress-bar">
                          <div 
                            className="progress-fill"
                            style={{ width: `${fileItem.progress}%` }}
                          />
                        </div>
                        <span className="progress-text">{fileItem.progress}%</span>
                      </div>
                    )}
                    
                    {fileItem.status === 'success' && (
                      <div className="success-info">
                        <CheckCircle className="text-green-500" size={16} />
                        <span>上传成功</span>
                        {fileItem.result && (
                          <div className="upload-result">
                            文档块数: {fileItem.result.documents_count}
                          </div>
                        )}
                      </div>
                    )}

                    {fileItem.status === 'error' && (
                      <div className="error-info">
                        <AlertCircle className="text-red-500" size={16} />
                        <span>{fileItem.error}</span>
                      </div>
                    )}

                    <button
                      className="remove-btn"
                      onClick={() => removeFile(fileItem.id)}
                    >
                      <X size={16} />
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Summary */}
            {(successCount > 0 || errorCount > 0) && (
              <div className="upload-summary">
                {successCount > 0 && (
                  <div className="summary-item success">
                    <CheckCircle size={16} />
                    <span>成功上传 {successCount} 个文件</span>
                  </div>
                )}
                {errorCount > 0 && (
                  <div className="summary-item error">
                    <AlertCircle size={16} />
                    <span>上传失败 {errorCount} 个文件</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      <style jsx>{`
        .upload-section {
          max-width: 800px;
          margin: 0 auto;
        }

        .upload-area {
          border: 2px dashed #cbd5e0;
          border-radius: 12px;
          padding: 3rem;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          background: white;
          margin-bottom: 2rem;
        }

        .upload-area:hover,
        .upload-area.drag-over {
          border-color: #3182ce;
          background-color: #f7fafc;
        }

        .upload-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1rem;
        }

        .upload-icon {
          color: #718096;
        }

        .upload-area:hover .upload-icon,
        .upload-area.drag-over .upload-icon {
          color: #3182ce;
        }

        .upload-content h3 {
          margin: 0;
          color: #2d3748;
        }

        .upload-content p {
          margin: 0;
          color: #718096;
          font-size: 0.9rem;
        }

        .file-list {
          background: white;
          border-radius: 12px;
          padding: 1.5rem;
          border: 1px solid #e2e8f0;
        }

        .file-list-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid #f7fafc;
        }

        .file-list-header h3 {
          margin: 0;
          color: #2d3748;
        }

        .file-actions {
          display: flex;
          gap: 0.75rem;
        }

        .files {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .file-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          background: #f7fafc;
        }

        .file-info {
          display: flex;
          align-items: center;
          gap: 1rem;
          flex: 1;
        }

        .file-icon {
          font-size: 1.5rem;
        }

        .file-details {
          flex: 1;
        }

        .file-name {
          font-weight: 500;
          color: #2d3748;
          margin-bottom: 0.25rem;
        }

        .file-size {
          font-size: 0.875rem;
          color: #718096;
        }

        .file-status {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .progress-container {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .progress-bar {
          width: 100px;
          height: 6px;
          background: #e2e8f0;
          border-radius: 3px;
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          background: #3182ce;
          transition: width 0.3s ease;
        }

        .progress-text {
          font-size: 0.875rem;
          color: #718096;
          min-width: 35px;
        }

        .success-info,
        .error-info {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.875rem;
        }

        .success-info {
          color: #2f855a;
        }

        .error-info {
          color: #c53030;
        }

        .upload-result {
          font-size: 0.75rem;
          color: #718096;
        }

        .remove-btn {
          background: none;
          border: none;
          color: #718096;
          cursor: pointer;
          padding: 0.25rem;
          border-radius: 4px;
          transition: all 0.2s;
        }

        .remove-btn:hover {
          color: #c53030;
          background: #fed7d7;
        }

        .upload-summary {
          margin-top: 1.5rem;
          padding-top: 1rem;
          border-top: 1px solid #f7fafc;
          display: flex;
          gap: 1rem;
        }

        .summary-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.875rem;
        }

        .summary-item.success {
          color: #2f855a;
        }

        .summary-item.error {
          color: #c53030;
        }

        @media (max-width: 768px) {
          .upload-area {
            padding: 2rem 1rem;
          }

          .file-list-header {
            flex-direction: column;
            gap: 1rem;
            align-items: stretch;
          }

          .file-actions {
            justify-content: center;
          }

          .file-item {
            flex-direction: column;
            align-items: stretch;
            gap: 1rem;
          }

          .file-status {
            justify-content: space-between;
          }
        }
      `}</style>
    </div>
  )
}

export default UploadPage