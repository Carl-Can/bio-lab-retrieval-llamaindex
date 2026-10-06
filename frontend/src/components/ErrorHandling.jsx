import React from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'

export const ErrorBoundary = class extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <AlertCircle size={48} className="error-icon" />
          <h2>出现了错误</h2>
          <p>页面遇到了意外错误，请刷新页面重试。</p>
          <button 
            className="retry-btn"
            onClick={() => window.location.reload()}
          >
            <RefreshCw size={16} />
            刷新页面
          </button>
          
          <style jsx>{`
            .error-boundary {
              display: flex;
              flex-direction: column;
              align-items: center;
              justify-content: center;
              min-height: 400px;
              padding: 2rem;
              text-align: center;
              color: #4a5568;
            }

            .error-icon {
              color: #e53e3e;
              margin-bottom: 1rem;
            }

            .error-boundary h2 {
              margin: 0 0 1rem 0;
              color: #2d3748;
            }

            .error-boundary p {
              margin: 0 0 2rem 0;
              max-width: 400px;
              line-height: 1.5;
            }

            .retry-btn {
              display: flex;
              align-items: center;
              gap: 0.5rem;
              padding: 0.75rem 1.5rem;
              background: #3182ce;
              color: white;
              border: none;
              border-radius: 8px;
              cursor: pointer;
              font-weight: 500;
              transition: background-color 0.2s;
            }

            .retry-btn:hover {
              background: #2c5aa0;
            }
          `}</style>
        </div>
      )
    }

    return this.props.children
  }
}

export const LoadingSpinner = ({ size = 32, text = '加载中...' }) => (
  <div className="loading-spinner">
    <div className="spinner" style={{ width: size, height: size }} />
    {text && <p>{text}</p>}
    
    <style jsx>{`
      .loading-spinner {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        color: #718096;
      }

      .spinner {
        border: 3px solid #e2e8f0;
        border-top: 3px solid #3182ce;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
      }

      .loading-spinner p {
        margin: 0;
        font-size: 0.9rem;
      }

      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `}</style>
  </div>
)

export const ErrorMessage = ({ error, onRetry, onDismiss }) => (
  <div className="error-message">
    <div className="error-content">
      <AlertCircle size={20} className="error-icon" />
      <span className="error-text">{error}</span>
    </div>
    <div className="error-actions">
      {onRetry && (
        <button className="retry-btn" onClick={onRetry}>
          重试
        </button>
      )}
      {onDismiss && (
        <button className="dismiss-btn" onClick={onDismiss}>
          ×
        </button>
      )}
    </div>
    
    <style jsx>{`
      .error-message {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem;
        background: #fed7d7;
        color: #c53030;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #feb2b2;
      }

      .error-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex: 1;
      }

      .error-icon {
        flex-shrink: 0;
      }

      .error-text {
        line-height: 1.4;
      }

      .error-actions {
        display: flex;
        gap: 0.5rem;
        align-items: center;
      }

      .retry-btn,
      .dismiss-btn {
        background: none;
        border: none;
        color: #c53030;
        cursor: pointer;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        transition: background-color 0.2s;
        font-size: 0.875rem;
      }

      .retry-btn:hover,
      .dismiss-btn:hover {
        background: rgba(197, 48, 48, 0.1);
      }

      .dismiss-btn {
        font-size: 1.25rem;
        font-weight: bold;
        padding: 0.25rem;
      }
    `}</style>
  </div>
)

export const SuccessMessage = ({ message, onDismiss }) => (
  <div className="success-message">
    <div className="success-content">
      <span className="success-text">{message}</span>
    </div>
    {onDismiss && (
      <button className="dismiss-btn" onClick={onDismiss}>
        ×
      </button>
    )}
    
    <style jsx>{`
      .success-message {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem;
        background: #c6f6d5;
        color: #2f855a;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #9ae6b4;
      }

      .success-content {
        flex: 1;
      }

      .success-text {
        line-height: 1.4;
      }

      .dismiss-btn {
        background: none;
        border: none;
        color: #2f855a;
        cursor: pointer;
        padding: 0.25rem;
        border-radius: 4px;
        transition: background-color 0.2s;
        font-size: 1.25rem;
        font-weight: bold;
      }

      .dismiss-btn:hover {
        background: rgba(47, 133, 90, 0.1);
      }
    `}</style>
  </div>
)