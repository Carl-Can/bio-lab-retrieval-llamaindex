import React, { useState, useRef, useEffect } from 'react'
import { Search, Send, Clock, FileText, Loader, AlertCircle, RotateCcw } from 'lucide-react'
import { apiService } from '../services/api'
import { formatTimestamp, highlightText } from '../utils/helpers'

const SearchPage = () => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [searchHistory, setSearchHistory] = useState([])
  const [topK, setTopK] = useState(3)
  const [useHybrid, setUseHybrid] = useState(true)
  const [useRerank, setUseRerank] = useState(true)
  const [retrieveOnly, setRetrieveOnly] = useState(false)
  const [topic, setTopic] = useState('')
  const [topics, setTopics] = useState([])
  const textareaRef = useRef(null)

  useEffect(() => {
    // Load search history from localStorage
    const savedHistory = localStorage.getItem('searchHistory')
    if (savedHistory) {
      setSearchHistory(JSON.parse(savedHistory))
    }
    // Load topic list for the filter dropdown
    apiService.getTopics()
      .then((data) => setTopics(data.topics || []))
      .catch(() => setTopics([]))
  }, [])

  const saveToHistory = (query, results) => {
    const historyItem = {
      id: Date.now(),
      query,
      timestamp: new Date().toISOString(),
      resultCount: results?.source_nodes?.length || 0
    }

    const newHistory = [historyItem, ...searchHistory.slice(0, 9)] // Keep last 10
    setSearchHistory(newHistory)
    localStorage.setItem('searchHistory', JSON.stringify(newHistory))
  }

  const handleSearch = async (searchQuery = query) => {
    if (!searchQuery.trim()) {
      setError('请输入查询内容')
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const searchResults = await apiService.queryDocuments(searchQuery, topK, {
        retrieval: useHybrid ? 'hybrid' : 'dense',
        rerank: useRerank ? 'bge' : 'none',
        topic: topic || undefined,
        generate: !retrieveOnly,
      })
      setResults(searchResults)
      saveToHistory(searchQuery, searchResults)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      handleSearch()
    }
  }

  const handleHistoryClick = (historyQuery) => {
    setQuery(historyQuery)
    handleSearch(historyQuery)
  }

  const clearHistory = () => {
    setSearchHistory([])
    localStorage.removeItem('searchHistory')
  }

  const clearResults = () => {
    setResults(null)
    setError(null)
  }

  const escapeHtml = (str) => {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
  }

  const safeHighlightText = (text, query) => {
    const escaped = escapeHtml(text)
    if (!query.trim()) return escaped
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    return escaped.replace(regex, '<mark>$1</mark>')
  }

  const exampleQueries = [
    "PCR实验的基本步骤是什么？",
    "如何提取DNA？",
    "蛋白质纯化的方法有哪些？",
    "细胞培养需要哪些条件？",
    "Western blot实验流程",
    "ELISA检测原理"
  ]

  return (
    <div className="page-container">
      <div className="search-header">
        <h1 className="page-title">智能检索</h1>
        <p className="page-description">
          使用自然语言查询实验方法、步骤、材料等信息
        </p>
      </div>

      {/* Search Input */}
      <div className="search-section">
        <div className="search-input-container">
          <div className="search-input-wrapper">
            <Search className="search-icon" size={20} />
            <textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="请输入您的问题，例如：PCR实验的基本步骤是什么？"
              className="search-input"
              rows="3"
            />
            <button 
              className="search-btn"
              onClick={() => handleSearch()}
              disabled={loading || !query.trim()}
            >
              {loading ? <Loader className="animate-spin" size={20} /> : <Send size={20} />}
            </button>
          </div>
          
          <div className="search-options">
            <label className="option-label">
              返回结果数量:
              <select 
                value={topK} 
                onChange={(e) => setTopK(Number(e.target.value))}
                className="option-select"
              >
                <option value={3}>3</option>
                <option value={5}>5</option>
                <option value={10}>10</option>
              </select>
            </label>

            <label className="option-label">
              专题:
              <select
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                className="option-select"
                title="限定在某个专题内检索；选错专题会召回不到"
              >
                <option value="">全部专题</option>
                {topics.map((t) => (
                  <option key={t.topic} value={t.topic}>
                    {t.topic} ({t.count})
                  </option>
                ))}
              </select>
            </label>

            <label className="option-toggle" title="向量 + BM25 关键词双通道，专有名词/缩写更不易漏">
              <input
                type="checkbox"
                checked={useHybrid}
                onChange={(e) => setUseHybrid(e.target.checked)}
              />
              混合检索
            </label>

            <label className="option-toggle" title="bge 精排候选，质量更好，每次约 +1s">
              <input
                type="checkbox"
                checked={useRerank}
                onChange={(e) => setUseRerank(e.target.checked)}
              />
              重排
            </label>

            <label className="option-toggle" title="只返回检索片段，不调用大模型生成">
              <input
                type="checkbox"
                checked={retrieveOnly}
                onChange={(e) => setRetrieveOnly(e.target.checked)}
              />
              仅检索
            </label>
            
            {(results || error) && (
              <button className="clear-btn" onClick={clearResults}>
                <RotateCcw size={16} />
                清空结果
              </button>
            )}
          </div>
        </div>

        {/* Example Queries */}
        {!results && !loading && (
          <div className="example-queries">
            <h3>示例查询</h3>
            <div className="example-grid">
              {exampleQueries.map((example, index) => (
                <button
                  key={index}
                  className="example-btn"
                  onClick={() => {
                    setQuery(example)
                    handleSearch(example)
                  }}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Search Results */}
      <div className="results-section">
        {loading && (
          <div className="loading-container">
            <Loader className="animate-spin" size={32} />
            <p>正在搜索相关信息...</p>
          </div>
        )}

        {error && (
          <div className="error-container">
            <AlertCircle size={24} />
            <p>{error}</p>
          </div>
        )}

        {results && (
          <div className="search-results">
            <div className="results-header">
              <h2>搜索结果</h2>
              <div className="results-info">
                找到 {results.source_nodes?.length || 0} 个相关结果
                {!results.response && (
                  <span className="retrieve-only-hint">（仅检索，未调用大模型生成）</span>
                )}
              </div>
            </div>

            {/* Main Response */}
            {results.response && (
              <div className="main-response">
                <h3>AI 回答</h3>
                <div className="response-content">
                  {results.response}
                </div>
              </div>
            )}

            {/* Source Documents */}
            {results.source_nodes && results.source_nodes.length > 0 && (
              <div className="source-documents">
                <h3>相关文档片段</h3>
                <div className="documents-list">
                  {results.source_nodes.map((node, index) => (
                    <div key={index} className="document-item">
                      <div className="document-header">
                        <FileText size={16} />
                        <span className="document-title">
                          {node.source_file ? `来源: ${node.source_file}` : `文档片段 ${index + 1}`}
                          {node.page_number && ` - 第${node.page_number}页`}
                        </span>
                        {typeof node.score === 'number' && (
                          <span className="relevance-score">
                            {node.score >= 0 && node.score <= 1
                              ? `相关度: ${Math.round(node.score * 100)}%`
                              : `分数: ${node.score.toFixed(2)}`}
                          </span>
                        )}
                      </div>
                      <div 
                        className="document-content"
                        dangerouslySetInnerHTML={{
                          __html: safeHighlightText(node.text, query)
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Search History Sidebar */}
      {searchHistory.length > 0 && (
        <div className="history-sidebar">
          <div className="history-header">
            <h3>搜索历史</h3>
            <button className="clear-history-btn" onClick={clearHistory}>
              清空
            </button>
          </div>
          <div className="history-list">
            {searchHistory.map((item) => (
              <div 
                key={item.id} 
                className="history-item"
                onClick={() => handleHistoryClick(item.query)}
              >
                <div className="history-query">{item.query}</div>
                <div className="history-meta">
                  <Clock size={12} />
                  <span>{formatTimestamp(item.timestamp)}</span>
                  <span>·</span>
                  <span>{item.resultCount} 个结果</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <style jsx>{`
        .search-header {
          text-align: center;
          margin-bottom: 2rem;
        }

        .search-section {
          max-width: 800px;
          margin: 0 auto 2rem;
        }

        .search-input-container {
          background: white;
          border-radius: 12px;
          padding: 1.5rem;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          border: 1px solid #e2e8f0;
        }

        .search-input-wrapper {
          position: relative;
          display: flex;
          align-items: flex-start;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .search-icon {
          color: #718096;
          margin-top: 0.75rem;
          flex-shrink: 0;
        }

        .search-input {
          flex: 1;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          padding: 0.75rem;
          font-size: 1rem;
          resize: vertical;
          min-height: 60px;
          font-family: inherit;
          transition: border-color 0.2s;
        }

        .search-input:focus {
          outline: none;
          border-color: #3182ce;
          box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
        }

        .search-btn {
          background: #3182ce;
          color: white;
          border: none;
          border-radius: 8px;
          padding: 0.75rem 1rem;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-weight: 500;
          margin-top: 0;
          height: fit-content;
        }

        .search-btn:hover:not(:disabled) {
          background: #2c5aa0;
        }

        .search-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .search-options {
          display: flex;
          flex-wrap: wrap;
          justify-content: flex-start;
          align-items: center;
          gap: 1rem;
        }

        .option-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
          color: #4a5568;
        }

        .option-toggle {
          display: flex;
          align-items: center;
          gap: 0.35rem;
          font-size: 0.9rem;
          color: #4a5568;
          cursor: pointer;
          user-select: none;
        }

        .option-toggle input {
          cursor: pointer;
        }

        .option-select {
          border: 1px solid #e2e8f0;
          border-radius: 4px;
          padding: 0.25rem 0.5rem;
          font-size: 0.9rem;
        }

        .clear-btn {
          background: #f7fafc;
          color: #4a5568;
          border: 1px solid #e2e8f0;
          border-radius: 6px;
          padding: 0.5rem 1rem;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.875rem;
          transition: all 0.2s;
        }

        .clear-btn:hover {
          background: #edf2f7;
        }

        .example-queries {
          margin-top: 2rem;
        }

        .example-queries h3 {
          margin-bottom: 1rem;
          color: #2d3748;
          font-size: 1.1rem;
        }

        .example-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 0.75rem;
        }

        .example-btn {
          background: white;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          padding: 0.75rem 1rem;
          text-align: left;
          cursor: pointer;
          transition: all 0.2s;
          font-size: 0.9rem;
          color: #4a5568;
        }

        .example-btn:hover {
          border-color: #3182ce;
          color: #3182ce;
          background: #f7fafc;
        }

        .results-section {
          max-width: 1000px;
          margin: 0 auto;
        }

        .loading-container,
        .error-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 3rem;
          text-align: center;
          color: #718096;
        }

        .error-container {
          color: #c53030;
          background: #fed7d7;
          border-radius: 8px;
        }

        .search-results {
          background: white;
          border-radius: 12px;
          padding: 2rem;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          border: 1px solid #e2e8f0;
        }

        .results-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid #f7fafc;
        }

        .results-header h2 {
          margin: 0;
          color: #2d3748;
        }

        .results-info {
          color: #718096;
          font-size: 0.9rem;
        }

        .retrieve-only-hint {
          margin-left: 0.5rem;
          color: #a0aec0;
        }

        .main-response {
          margin-bottom: 2rem;
        }

        .main-response h3 {
          margin-bottom: 1rem;
          color: #2d3748;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .response-content {
          background: #f7fafc;
          border-left: 4px solid #3182ce;
          padding: 1.5rem;
          border-radius: 0 8px 8px 0;
          line-height: 1.6;
          color: #2d3748;
          white-space: pre-wrap;
        }

        .source-documents h3 {
          margin-bottom: 1rem;
          color: #2d3748;
        }

        .documents-list {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .document-item {
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          overflow: hidden;
        }

        .document-header {
          background: #f7fafc;
          padding: 1rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          border-bottom: 1px solid #e2e8f0;
        }

        .document-title {
          font-weight: 500;
          color: #2d3748;
          flex: 1;
        }

        .relevance-score {
          background: #e6fffa;
          color: #2f855a;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
          font-weight: 500;
        }

        .document-content {
          padding: 1.5rem;
          line-height: 1.6;
          color: #4a5568;
        }

        .document-content :global(mark) {
          background: #fef5e7;
          color: #d69e2e;
          padding: 0.125rem 0.25rem;
          border-radius: 3px;
        }

        .history-sidebar {
          position: fixed;
          top: 50%;
          right: 1rem;
          transform: translateY(-50%);
          width: 300px;
          max-height: 500px;
          background: white;
          border-radius: 12px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
          border: 1px solid #e2e8f0;
          overflow: hidden;
          z-index: 100;
        }

        .history-header {
          background: #f7fafc;
          padding: 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          border-bottom: 1px solid #e2e8f0;
        }

        .history-header h3 {
          margin: 0;
          font-size: 1rem;
          color: #2d3748;
        }

        .clear-history-btn {
          background: none;
          border: none;
          color: #718096;
          cursor: pointer;
          font-size: 0.875rem;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          transition: all 0.2s;
        }

        .clear-history-btn:hover {
          color: #c53030;
          background: #fed7d7;
        }

        .history-list {
          max-height: 400px;
          overflow-y: auto;
        }

        .history-item {
          padding: 1rem;
          border-bottom: 1px solid #f7fafc;
          cursor: pointer;
          transition: background-color 0.2s;
        }

        .history-item:hover {
          background: #f7fafc;
        }

        .history-item:last-child {
          border-bottom: none;
        }

        .history-query {
          font-size: 0.9rem;
          color: #2d3748;
          margin-bottom: 0.5rem;
          line-height: 1.4;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        .history-meta {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          font-size: 0.75rem;
          color: #718096;
        }

        @media (max-width: 1200px) {
          .history-sidebar {
            display: none;
          }
        }

        @media (max-width: 768px) {
          .search-options {
            flex-direction: column;
            align-items: stretch;
            gap: 0.75rem;
          }

          .example-grid {
            grid-template-columns: 1fr;
          }

          .search-input-wrapper {
            flex-direction: column;
            align-items: stretch;
          }

          .search-icon {
            display: none;
          }

          .search-btn {
            align-self: flex-end;
            width: fit-content;
          }

          .results-header {
            flex-direction: column;
            align-items: stretch;
            gap: 0.5rem;
          }
        }
      `}</style>
    </div>
  )
}

export default SearchPage