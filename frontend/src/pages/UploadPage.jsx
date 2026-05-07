import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Play, CheckCircle } from 'lucide-react';

const API_BASE = "/api";

export default function UploadPage() {
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [runId, setRunId] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const uploadImages = async () => {
    if (files.length === 0) return;
    setIsUploading(true);
    
    const formData = new FormData();
    files.forEach(file => formData.append("files", file));

    try {
      const resp = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData
      });
      const data = await resp.json();
      setRunId(data.run_id);
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div>
      <h1 className="title">Industrial Data Upload</h1>
      
      <div className="card" style={{borderStyle: 'dashed', textAlign: 'center', padding: '4rem'}}>
        <Upload size={48} color="var(--accent-primary)" style={{marginBottom: '1rem'}} />
        <p style={{color: 'var(--text-dim)', marginBottom: '1.5rem'}}>
          Drag and drop industrial images or click to select
        </p>
        <input 
          type="file" 
          multiple 
          onChange={handleFileChange} 
          id="file-input"
          style={{display: 'none'}}
        />
        <label htmlFor="file-input" className="btn btn-secondary">
          SELECT FILES
        </label>
        
        {files.length > 0 && (
          <div style={{marginTop: '2rem', textAlign: 'left'}}>
            <p style={{fontSize: '0.9rem', fontWeight: 600}}>Queue: {files.length} images</p>
            {!runId && !isUploading && (
              <button className="btn" style={{marginTop: '1rem', width: '100%'}} onClick={uploadImages}>
                START UPLOAD & GENERATE RUN
              </button>
            )}
          </div>
        )}

        {isUploading && <p style={{marginTop: '1rem'}}>Uploading to pod...</p>}

        {runId && (
          <div style={{marginTop: '2rem', padding: '1rem', background: 'rgba(0,255,136,0.1)', borderRadius: '8px'}}>
            <CheckCircle color="var(--success)" style={{marginBottom: '0.5rem'}} />
            <p>Run <strong>{runId}</strong> initialized successfully.</p>
            <button className="btn" style={{marginTop: '1rem'}} onClick={() => navigate(`/annotate/${runId}`)}>
              PROCEED TO ANNOTATION
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
