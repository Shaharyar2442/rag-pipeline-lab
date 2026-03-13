import React, { useState, useEffect } from 'react'

const API_BASE = 'http://localhost:5000/api'

const PIPELINES = [
  { id: 'rag_fusion', name: 'RAG Fusion', icon: '🔀', desc: 'Multi-query + Reciprocal Rank Fusion' },
  { id: 'hyde', name: 'HyDE', icon: '📝', desc: 'Hypothetical Document Embedding' },
  { id: 'crag', name: 'CRAG', icon: '🔍', desc: 'Corrective RAG with confidence gating' },
  { id: 'graph_rag', name: 'Graph RAG', icon: '🕸️', desc: 'Graph-augmented entity retrieval' },
]

export default function App() {
  const [query, setQuery] = useState('')
  const [selectedPipeline, setSelectedPipeline] = useState('rag_fusion')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [samples, setSamples] = useState([])
  const [expandedChunk, setExpandedChunk] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/samples`)
      .then(r => r.json())
      .then(data => setSamples(data.samples || []))
      .catch(() => {})
  }, [])

  const runQuery = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const resp = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), pipeline: selectedPipeline }),
      })
      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.error || 'Request failed')
      }
      const data = await resp.json()
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      runQuery()
    }
  }

  return (
    <div style={styles.app}>
      <style>{globalCSS}</style>

      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerGlow}></div>
        <h1 style={styles.title}>
          <span style={styles.titleIcon}>⚡</span>
          RAG Pipeline Explorer
        </h1>
        <p style={styles.subtitle}>Compare advanced RAG strategies on the CRAG dataset</p>
      </header>

      <main style={styles.main}>
        {/* Query Section */}
        <section style={styles.section}>
          <div style={styles.queryContainer}>
            <div style={styles.inputWrapper}>
              <input
                id="query-input"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question... e.g., Who directed Inception?"
                style={styles.input}
              />
              <button
                id="run-btn"
                onClick={runQuery}
                disabled={loading || !query.trim()}
                style={{
                  ...styles.runButton,
                  ...(loading ? styles.runButtonDisabled : {}),
                }}
              >
                {loading ? (
                  <span style={styles.spinner}>⏳</span>
                ) : (
                  '→'
                )}
              </button>
            </div>

            {/* Sample queries dropdown */}
            {samples.length > 0 && (
              <div style={styles.samplesRow}>
                <span style={styles.samplesLabel}>Try:</span>
                <div style={styles.sampleChips}>
                  {samples.slice(0, 6).map((s, i) => (
                    <button
                      key={i}
                      onClick={() => setQuery(s.query)}
                      style={styles.sampleChip}
                      title={s.query}
                    >
                      {s.query.length > 40 ? s.query.slice(0, 40) + '…' : s.query}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Pipeline Selector */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Select Pipeline</h2>
          <div style={styles.pipelineGrid}>
            {PIPELINES.map((p) => (
              <button
                key={p.id}
                id={`pipeline-${p.id}`}
                onClick={() => setSelectedPipeline(p.id)}
                style={{
                  ...styles.pipelineCard,
                  ...(selectedPipeline === p.id ? styles.pipelineCardActive : {}),
                }}
              >
                <span style={styles.pipelineIcon}>{p.icon}</span>
                <span style={styles.pipelineName}>{p.name}</span>
                <span style={styles.pipelineDesc}>{p.desc}</span>
              </button>
            ))}
          </div>
        </section>

        {/* Error */}
        {error && (
          <div style={styles.errorBox}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={styles.loadingBox}>
            <div style={styles.loadingSpinner}></div>
            <p style={styles.loadingText}>
              Running {PIPELINES.find(p => p.id === selectedPipeline)?.name}...
            </p>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <section style={styles.resultsSection}>
            {/* Answer */}
            <div style={styles.answerCard}>
              <h2 style={styles.answerTitle}>
                💡 Generated Answer
                <span style={styles.pipelineBadge}>
                  {PIPELINES.find(p => p.id === result.pipeline)?.name || result.pipeline}
                </span>
              </h2>
              <p style={styles.answerText}>{result.answer}</p>

              {/* Pipeline-specific info */}
              {result.query_variants && (
                <div style={styles.metaBox}>
                  <h4 style={styles.metaTitle}>Query Variants Generated:</h4>
                  <ul style={styles.metaList}>
                    {result.query_variants.map((v, i) => (
                      <li key={i} style={styles.metaItem}>{v}</li>
                    ))}
                  </ul>
                </div>
              )}

              {result.hypothetical_document && (
                <div style={styles.metaBox}>
                  <h4 style={styles.metaTitle}>Hypothetical Document:</h4>
                  <p style={styles.metaText}>{result.hypothetical_document}</p>
                </div>
              )}

              {result.confidence_level && (
                <div style={styles.metaBox}>
                  <h4 style={styles.metaTitle}>
                    Confidence: <span style={{
                      color: result.confidence_level === 'high' ? '#4ade80' :
                             result.confidence_level === 'medium' ? '#fbbf24' : '#f87171'
                    }}>{result.confidence_level.toUpperCase()}</span>
                    {result.mean_confidence !== undefined && (
                      <span style={styles.confidenceScore}> ({(result.mean_confidence * 100).toFixed(1)}%)</span>
                    )}
                  </h4>
                  {result.note && <p style={styles.metaText}>{result.note}</p>}
                </div>
              )}

              {result.graph_info && (
                <div style={styles.metaBox}>
                  <h4 style={styles.metaTitle}>Graph Info:</h4>
                  <p style={styles.metaText}>
                    Query entities: {result.query_entities?.join(', ') || 'none'}<br/>
                    Seed chunks: {result.graph_info.seed_chunks} | 
                    Expanded: {result.graph_info.expanded_chunks}
                    {result.graph_info.note && <><br/>{result.graph_info.note}</>}
                  </p>
                </div>
              )}
            </div>

            {/* Retrieved Chunks */}
            {result.retrieved_chunks && result.retrieved_chunks.length > 0 && (
              <div style={styles.chunksSection}>
                <h2 style={styles.chunksTitle}>
                  📄 Retrieved Chunks ({result.retrieved_chunks.length})
                </h2>
                <div style={styles.chunksList}>
                  {result.retrieved_chunks.map((chunk, i) => (
                    <div
                      key={i}
                      style={styles.chunkCard}
                      onClick={() => setExpandedChunk(expandedChunk === i ? null : i)}
                    >
                      <div style={styles.chunkHeader}>
                        <span style={styles.chunkIndex}>#{i + 1}</span>
                        <span style={styles.chunkScore}>
                          Score: {typeof chunk.score === 'number' ? chunk.score.toFixed(4) : chunk.score}
                        </span>
                        {chunk.confidence !== undefined && (
                          <span style={{
                            ...styles.chunkConfidence,
                            color: chunk.confidence >= 0.65 ? '#4ade80' :
                                   chunk.confidence >= 0.35 ? '#fbbf24' : '#f87171'
                          }}>
                            Confidence: {(chunk.confidence * 100).toFixed(1)}%
                          </span>
                        )}
                        <span style={styles.expandIcon}>
                          {expandedChunk === i ? '▼' : '▶'}
                        </span>
                      </div>
                      <p style={styles.chunkText}>
                        {expandedChunk === i
                          ? chunk.text
                          : (chunk.text?.length > 200
                              ? chunk.text.slice(0, 200) + '…'
                              : chunk.text)}
                      </p>
                      {chunk.source && (
                        <div style={styles.chunkSource}>
                          📌 {chunk.source}
                          {chunk.url && (
                            <a href={chunk.url} target="_blank" rel="noreferrer" style={styles.chunkUrl}>
                              ↗
                            </a>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      <footer style={styles.footer}>
        <p>RAG in the Wild — CRAG Case Study • CS-4015 Agentic AI</p>
      </footer>
    </div>
  )
}

// ─── Global CSS for animations ───
const globalCSS = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0a0a0f;
    color: #e4e4e7;
    min-height: 100vh;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  @keyframes glow {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.6; }
  }
`

// ─── Inline styles ───
const styles = {
  app: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    position: 'relative',
    padding: '48px 24px 32px',
    textAlign: 'center',
    borderBottom: '1px solid rgba(139, 92, 246, 0.2)',
    background: 'linear-gradient(180deg, rgba(139, 92, 246, 0.08) 0%, transparent 100%)',
    overflow: 'hidden',
  },
  headerGlow: {
    position: 'absolute',
    top: '-50%',
    left: '50%',
    transform: 'translateX(-50%)',
    width: '600px',
    height: '300px',
    background: 'radial-gradient(ellipse, rgba(139, 92, 246, 0.15), transparent 70%)',
    animation: 'glow 4s ease-in-out infinite',
    pointerEvents: 'none',
  },
  title: {
    fontSize: '2.2rem',
    fontWeight: 700,
    background: 'linear-gradient(135deg, #a78bfa, #818cf8, #6366f1)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    marginBottom: '8px',
    position: 'relative',
  },
  titleIcon: { marginRight: '10px' },
  subtitle: {
    color: '#71717a',
    fontSize: '0.95rem',
    fontWeight: 400,
    position: 'relative',
  },
  main: {
    flex: 1,
    maxWidth: '960px',
    width: '100%',
    margin: '0 auto',
    padding: '32px 24px',
  },
  section: {
    marginBottom: '32px',
  },
  sectionTitle: {
    fontSize: '1rem',
    fontWeight: 600,
    color: '#a1a1aa',
    marginBottom: '14px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  queryContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  inputWrapper: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
  },
  input: {
    flex: 1,
    padding: '14px 18px',
    fontSize: '1rem',
    background: 'rgba(24, 24, 27, 0.8)',
    border: '1px solid rgba(63, 63, 70, 0.6)',
    borderRadius: '12px',
    color: '#e4e4e7',
    outline: 'none',
    transition: 'border-color 0.2s, box-shadow 0.2s',
    backdropFilter: 'blur(8px)',
  },
  runButton: {
    padding: '14px 22px',
    fontSize: '1.2rem',
    fontWeight: 700,
    background: 'linear-gradient(135deg, #7c3aed, #6366f1)',
    border: 'none',
    borderRadius: '12px',
    color: '#fff',
    cursor: 'pointer',
    transition: 'transform 0.15s, box-shadow 0.2s',
    boxShadow: '0 0 20px rgba(99, 102, 241, 0.3)',
  },
  runButtonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  spinner: {
    display: 'inline-block',
    animation: 'spin 1s linear infinite',
  },
  samplesRow: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '8px',
    flexWrap: 'wrap',
  },
  samplesLabel: {
    color: '#71717a',
    fontSize: '0.8rem',
    fontWeight: 500,
    paddingTop: '6px',
    flexShrink: 0,
  },
  sampleChips: {
    display: 'flex',
    gap: '6px',
    flexWrap: 'wrap',
    flex: 1,
  },
  sampleChip: {
    padding: '5px 12px',
    fontSize: '0.75rem',
    background: 'rgba(39, 39, 42, 0.8)',
    border: '1px solid rgba(63, 63, 70, 0.4)',
    borderRadius: '20px',
    color: '#a1a1aa',
    cursor: 'pointer',
    transition: 'all 0.2s',
    whiteSpace: 'nowrap',
  },
  pipelineGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '12px',
  },
  pipelineCard: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '6px',
    padding: '18px 14px',
    background: 'rgba(24, 24, 27, 0.6)',
    border: '1px solid rgba(63, 63, 70, 0.4)',
    borderRadius: '14px',
    cursor: 'pointer',
    transition: 'all 0.25s ease',
    backdropFilter: 'blur(8px)',
    textAlign: 'center',
  },
  pipelineCardActive: {
    background: 'rgba(99, 102, 241, 0.12)',
    borderColor: 'rgba(99, 102, 241, 0.6)',
    boxShadow: '0 0 24px rgba(99, 102, 241, 0.15)',
  },
  pipelineIcon: { fontSize: '1.6rem' },
  pipelineName: {
    fontSize: '0.95rem',
    fontWeight: 600,
    color: '#e4e4e7',
  },
  pipelineDesc: {
    fontSize: '0.72rem',
    color: '#71717a',
    lineHeight: 1.3,
  },
  errorBox: {
    padding: '14px 18px',
    background: 'rgba(239, 68, 68, 0.1)',
    border: '1px solid rgba(239, 68, 68, 0.3)',
    borderRadius: '10px',
    color: '#fca5a5',
    fontSize: '0.9rem',
    marginBottom: '20px',
    animation: 'fadeIn 0.3s ease',
  },
  loadingBox: {
    textAlign: 'center',
    padding: '48px 20px',
    animation: 'fadeIn 0.3s ease',
  },
  loadingSpinner: {
    width: '40px',
    height: '40px',
    border: '3px solid rgba(99, 102, 241, 0.2)',
    borderTopColor: '#6366f1',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
    margin: '0 auto 16px',
  },
  loadingText: {
    color: '#71717a',
    fontSize: '0.95rem',
    animation: 'pulse 1.5s ease infinite',
  },
  resultsSection: {
    animation: 'fadeIn 0.4s ease',
  },
  answerCard: {
    padding: '24px',
    background: 'linear-gradient(135deg, rgba(24, 24, 27, 0.9), rgba(30, 30, 36, 0.9))',
    border: '1px solid rgba(99, 102, 241, 0.25)',
    borderRadius: '16px',
    marginBottom: '24px',
    backdropFilter: 'blur(12px)',
    boxShadow: '0 4px 30px rgba(0, 0, 0, 0.2)',
  },
  answerTitle: {
    fontSize: '1.1rem',
    fontWeight: 600,
    color: '#a78bfa',
    marginBottom: '14px',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    flexWrap: 'wrap',
  },
  pipelineBadge: {
    fontSize: '0.7rem',
    fontWeight: 500,
    padding: '3px 10px',
    background: 'rgba(99, 102, 241, 0.15)',
    border: '1px solid rgba(99, 102, 241, 0.3)',
    borderRadius: '20px',
    color: '#818cf8',
  },
  answerText: {
    fontSize: '1rem',
    lineHeight: 1.7,
    color: '#d4d4d8',
    whiteSpace: 'pre-wrap',
  },
  metaBox: {
    marginTop: '16px',
    padding: '12px 16px',
    background: 'rgba(39, 39, 42, 0.5)',
    borderRadius: '10px',
    border: '1px solid rgba(63, 63, 70, 0.3)',
  },
  metaTitle: {
    fontSize: '0.85rem',
    fontWeight: 600,
    color: '#a1a1aa',
    marginBottom: '6px',
  },
  metaList: {
    listStyle: 'none',
    padding: 0,
  },
  metaItem: {
    fontSize: '0.82rem',
    color: '#71717a',
    padding: '3px 0',
    borderBottom: '1px solid rgba(63, 63, 70, 0.2)',
  },
  metaText: {
    fontSize: '0.82rem',
    color: '#71717a',
    lineHeight: 1.5,
  },
  confidenceScore: {
    fontWeight: 400,
    fontSize: '0.8rem',
  },
  chunksSection: {
    marginBottom: '24px',
  },
  chunksTitle: {
    fontSize: '1.05rem',
    fontWeight: 600,
    color: '#a1a1aa',
    marginBottom: '14px',
  },
  chunksList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  chunkCard: {
    padding: '16px',
    background: 'rgba(24, 24, 27, 0.7)',
    border: '1px solid rgba(63, 63, 70, 0.35)',
    borderRadius: '12px',
    cursor: 'pointer',
    transition: 'border-color 0.2s, background 0.2s',
  },
  chunkHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '8px',
    flexWrap: 'wrap',
  },
  chunkIndex: {
    fontSize: '0.8rem',
    fontWeight: 700,
    color: '#6366f1',
    background: 'rgba(99, 102, 241, 0.1)',
    padding: '2px 8px',
    borderRadius: '6px',
  },
  chunkScore: {
    fontSize: '0.78rem',
    color: '#4ade80',
    fontWeight: 500,
    fontFamily: 'monospace',
  },
  chunkConfidence: {
    fontSize: '0.78rem',
    fontWeight: 500,
    fontFamily: 'monospace',
  },
  expandIcon: {
    marginLeft: 'auto',
    fontSize: '0.7rem',
    color: '#52525b',
  },
  chunkText: {
    fontSize: '0.85rem',
    color: '#a1a1aa',
    lineHeight: 1.6,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  chunkSource: {
    marginTop: '8px',
    fontSize: '0.75rem',
    color: '#52525b',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  chunkUrl: {
    color: '#6366f1',
    textDecoration: 'none',
    fontSize: '0.85rem',
  },
  footer: {
    textAlign: 'center',
    padding: '24px',
    color: '#3f3f46',
    fontSize: '0.8rem',
    borderTop: '1px solid rgba(63, 63, 70, 0.2)',
  },
}
