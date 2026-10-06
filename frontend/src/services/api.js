import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error)
    
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response
      throw new Error(data.detail || `Server error: ${status}`)
    } else if (error.request) {
      // Request was made but no response
      throw new Error('网络连接失败，请检查服务器是否运行')
    } else {
      // Something else happened
      throw new Error(error.message || '请求失败')
    }
  }
)

export const apiService = {
  // Get system info
  async getSystemInfo() {
    const response = await api.get('/')
    return response.data
  },

  // Get stats
  async getStats() {
    const response = await api.get('/stats')
    return response.data
  },

  // Get all documents
  async getDocuments() {
    const response = await api.get('/documents')
    return response.data
  },

  // Upload document
  async uploadDocument(file, onProgress) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    })

    return response.data
  },

  // Query documents
  // options: { retrieval: 'dense'|'hybrid', rerank: 'none'|'bge'|'jev', topic: string, generate: boolean }
  async queryDocuments(query, topK = 3, options = {}) {
    const { retrieval, rerank, topic, generate = true } = options
    const params = { q: query, top_k: topK, generate }
    if (retrieval) params.retrieval = retrieval
    if (rerank) params.rerank = rerank
    if (topic) params.topic = topic

    const response = await api.get('/query', { params })
    return response.data
  },

  // Get topics (docs 一级目录) with document counts
  async getTopics() {
    const response = await api.get('/topics')
    return response.data
  },

  // Get document by ID
  async getDocument(docId) {
    const response = await api.get(`/docs/${docId}`)
    return response.data
  },

  // Health check
  async healthCheck() {
    try {
      const response = await api.get('/', { timeout: 5000 })
      return { status: 'ok', data: response.data }
    } catch (error) {
      return { status: 'error', error: error.message }
    }
  },

  // Initialize index from docs folder
  async initializeIndex() {
    const response = await api.post('/initialize')
    return response.data
  },
}

export default apiService