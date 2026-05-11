import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Activity, BarChart3, Image as ImageIcon } from 'lucide-react';

const API_BASE = "/api";

export default function ResultsPage() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [metrics, setMetrics] = useState(null);
  const [status, setStatus] = useState("Checking training status...");
  const [isTraining, setIsTraining] = useState(true);
  const [runsHistory, setRunsHistory] = useState([]);

  useEffect(() => {
    if (runId) return;
    const fetchRuns = async () => {
      try {
        const resp = await fetch(`${API_BASE}/runs`);
        const data = await resp.json();
        setRunsHistory(data.runs || []);
      } catch (e) {
        console.error("Failed to fetch runs", e);
      }
    };
    fetchRuns();
  }, [runId]);

  useEffect(() => {
    if (!runId) return;

    const checkStatus = async () => {
      // In a real app, we'd poll the task ID returned by /train
      // For this MVP, we'll try to fetch results every 5 seconds
      try {
        const resp = await fetch(`${API_BASE}/results/${runId}`);
        if (resp.ok) {
          const data = await resp.json();
          setMetrics(data.metrics);
          setIsTraining(false);
          return true;
        }
      } catch (e) {}
      return false;
    };

    const interval = setInterval(async () => {
      const done = await checkStatus();
      if (done) clearInterval(interval);
    }, 5000);

    checkStatus();
    return () => clearInterval(interval);
  }, [runId]);

  if (!runId) {
    return (
      <div className="app-container">
        <h1 className="title">Select a Run to View Results</h1>
        {runsHistory.length === 0 ? (
          <p style={{color: 'var(--text-dim)'}}>No runs found. Please go to UPLOAD and start a run.</p>
        ) : (
          <div style={{display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '2rem'}}>
            {runsHistory.map(id => (
              <button key={id} className="btn btn-secondary" style={{textAlign: 'left', display: 'flex', justifyContent: 'space-between', padding: '1rem'}} onClick={() => navigate(`/results/${id}`)}>
                <span>Run ID: <strong>{id}</strong></span>
                <span>View Results ➔</span>
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }


  return (
    <div>
      <h1 className="title">Model Analysis</h1>

      {isTraining ? (
        <div className="card" style={{textAlign: 'center', padding: '4rem'}}>
          <Activity className="spin" size={48} color="var(--accent-primary)" style={{marginBottom: '1rem'}} />
          <h2>Training Pipeline Active</h2>
          <p style={{color: 'var(--text-dim)'}}>Optimizing weights on RunPod GPU. This usually takes 2-3 minutes.</p>
        </div>
      ) : (
        <div style={{display: 'flex', flexDirection: 'column', gap: '2rem'}}>
          <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem'}}>
            <div className="card" style={{textAlign: 'center'}}>
              <p style={{color: 'var(--text-dim)', fontSize: '0.8rem', fontWeight: 600}}>mAP @ 0.5:0.95</p>
              <h2 style={{fontSize: '2.5rem', color: 'var(--accent-primary)'}}>{(metrics?.mAP_50_95 || 0).toFixed(3)}</h2>
            </div>
            <div className="card" style={{textAlign: 'center'}}>
              <p style={{color: 'var(--text-dim)', fontSize: '0.8rem', fontWeight: 600}}>mAP @ 0.5</p>
              <h2 style={{fontSize: '2.5rem', color: 'var(--accent-secondary)'}}>{(metrics?.mAP_50 || 0).toFixed(3)}</h2>
            </div>
            <div className="card" style={{textAlign: 'center'}}>
              <p style={{color: 'var(--text-dim)', fontSize: '0.8rem', fontWeight: 600}}>Recall (AR@100)</p>
              <h2 style={{fontSize: '2.5rem', color: 'var(--success)'}}>{(metrics?.AR_100 || 0).toFixed(3)}</h2>
            </div>
          </div>

          <div className="card">
            <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem'}}>
              <BarChart3 size={20} color="var(--accent-primary)" />
              <h3 style={{fontSize: '1.2rem'}}>Inference Visualizer</h3>
            </div>
            <p style={{color: 'var(--text-dim)', marginBottom: '1rem'}}>
              Predictions have been saved to the run directory. (Detailed gallery visualization coming in next update).
            </p>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .spin {
          animation: spin 2s linear infinite;
        }
      `}</style>
    </div>
  );
}
