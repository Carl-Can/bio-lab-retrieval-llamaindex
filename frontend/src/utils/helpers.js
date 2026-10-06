// Utility functions for the application

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

export const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp)
  const now = new Date()
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return '刚刚'
  if (diffMins < 60) return `${diffMins}分钟前`
  if (diffHours < 24) return `${diffHours}小时前`
  if (diffDays < 7) return `${diffDays}天前`
  return date.toLocaleDateString('zh-CN')
}

export const getFileIcon = (filename) => {
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

export const validateFile = (file) => {
  const maxSize = 10 * 1024 * 1024 // 10MB
  const allowedTypes = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/markdown',
    'text/plain'
  ]

  const errors = []

  if (!allowedTypes.includes(file.type)) {
    errors.push('文件类型不支持。请上传 PDF, Word, Markdown 或 TXT 文件。')
  }

  if (file.size > maxSize) {
    errors.push(`文件大小超过限制。最大允许 ${formatFileSize(maxSize)}。`)
  }

  return {
    isValid: errors.length === 0,
    errors
  }
}

export const highlightText = (text, query) => {
  if (!query.trim()) return text
  
  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
  return text.replace(regex, '<mark>$1</mark>')
}

export const debounce = (func, wait) => {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}

export const truncateText = (text, maxLength = 100) => {
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

// Local storage helpers
export const storage = {
  get(key) {
    try {
      const item = localStorage.getItem(key)
      return item ? JSON.parse(item) : null
    } catch (error) {
      console.error('Error reading from localStorage:', error)
      return null
    }
  },

  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value))
      return true
    } catch (error) {
      console.error('Error writing to localStorage:', error)
      return false
    }
  },

  remove(key) {
    try {
      localStorage.removeItem(key)
      return true
    } catch (error) {
      console.error('Error removing from localStorage:', error)
      return false
    }
  },

  clear() {
    try {
      localStorage.clear()
      return true
    } catch (error) {
      console.error('Error clearing localStorage:', error)
      return false
    }
  }
}

// Search history management
export const searchHistory = {
  getHistory() {
    return storage.get('searchHistory') || []
  },

  addSearch(query, resultCount = 0) {
    const history = this.getHistory()
    const newItem = {
      id: Date.now(),
      query,
      timestamp: new Date().toISOString(),
      resultCount
    }

    const updatedHistory = [newItem, ...history.slice(0, 9)] // Keep last 10
    storage.set('searchHistory', updatedHistory)
    return updatedHistory
  },

  clearHistory() {
    storage.remove('searchHistory')
  }
}

// Upload history management
export const uploadHistory = {
  getHistory() {
    return storage.get('uploadHistory') || []
  },

  addUpload(fileInfo) {
    const history = this.getHistory()
    const newItem = {
      ...fileInfo,
      id: fileInfo.file_id || Date.now(),
      upload_time: new Date().toISOString()
    }

    const updatedHistory = [newItem, ...history]
    storage.set('uploadHistory', updatedHistory)
    return updatedHistory
  },

  removeUpload(fileId) {
    const history = this.getHistory()
    const updatedHistory = history.filter(item => item.id !== fileId)
    storage.set('uploadHistory', updatedHistory)
    return updatedHistory
  },

  clearHistory() {
    storage.remove('uploadHistory')
  }
}

// API error handling
export const handleApiError = (error) => {
  if (error.response) {
    // Server responded with error status
    const { status, data } = error.response
    if (status === 404) {
      return '请求的资源不存在'
    } else if (status === 500) {
      return '服务器内部错误，请稍后重试'
    } else if (status === 400) {
      return data.detail || '请求参数错误'
    } else {
      return data.detail || `服务器错误: ${status}`
    }
  } else if (error.request) {
    // Request was made but no response
    return '网络连接失败，请检查服务器是否运行'
  } else {
    // Something else happened
    return error.message || '请求失败'
  }
}

// Copy to clipboard
export const copyToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch (error) {
    // Fallback for older browsers
    const textArea = document.createElement('textarea')
    textArea.value = text
    document.body.appendChild(textArea)
    textArea.select()
    try {
      document.execCommand('copy')
      return true
    } catch (err) {
      console.error('Failed to copy text:', err)
      return false
    } finally {
      document.body.removeChild(textArea)
    }
  }
}