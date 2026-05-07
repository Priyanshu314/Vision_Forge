import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Save, ChevronLeft, ChevronRight } from 'lucide-react';

const API_BASE = "/api";

export default function AnnotationPage() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [annotations, setAnnotations] = useState({}); // {image_id: [bboxes]}
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentBox, setCurrentBox] = useState(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  // Fetch images for the run
  useEffect(() => {
    if (!runId) return;
    // In a real app, we'd have a route to list images in a run
    // For this MVP, we'll assume a fixed number or handle it via sampling
    const fetchImages = async () => {
       const resp = await fetch(`${API_BASE}/sample/${runId}?limit=20`);
       const data = await resp.json();
       setImages(data.samples || []);
    };
    fetchImages();
  }, [runId]);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw existing boxes
    const currentAnns = annotations[images[currentIndex]] || [];
    ctx.strokeStyle = '#00f2ff';
    ctx.lineWidth = 2;
    currentAnns.forEach(box => {
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.fillStyle = 'rgba(0, 242, 255, 0.1)';
      ctx.fillRect(box.x, box.y, box.w, box.h);
    });

    // Draw current box
    if (currentBox) {
      ctx.strokeStyle = '#ffb800';
      ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
    }
  };

  useEffect(draw, [currentIndex, annotations, currentBox, images]);

  const handleMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setStartPos({ x, y });
    setIsDrawing(true);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCurrentBox({
      x: Math.min(x, startPos.x),
      y: Math.min(y, startPos.y),
      w: Math.abs(x - startPos.x),
      h: Math.abs(y - startPos.y)
    });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentBox) return;
    const imgId = images[currentIndex];
    setAnnotations(prev => ({
      ...prev,
      [imgId]: [...(prev[imgId] || []), currentBox]
    }));
    setIsDrawing(false);
    setCurrentBox(null);
  };

  const saveAll = async () => {
    // Convert to COCO format
    const coco = {
      images: images.map((name, i) => ({ id: i, file_name: name })),
      categories: [{ id: 1, name: "defect" }],
      annotations: []
    };

    let annId = 0;
    images.forEach((name, imgIdx) => {
      const boxes = annotations[name] || [];
      boxes.forEach(box => {
        coco.annotations.append({
          id: annId++,
          image_id: imgIdx,
          category_id: 1,
          bbox: [box.x, box.y, box.w, box.h]
        });
      });
    });

    try {
      await fetch(`${API_BASE}/annotate/${runId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(coco)
      });
      alert("Annotations saved! Starting training...");
      await fetch(`${API_BASE}/train/${runId}`, { method: "POST" });
      navigate(`/results/${runId}`);
    } catch (err) {
      alert("Save failed: " + err.message);
    }
  };

  if (!runId) return <div className="app-container">Please start a run first.</div>;
  if (images.length === 0) return <div className="app-container">Loading images...</div>;

  return (
    <div>
      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem'}}>
        <h1 className="title">Annotate Defects ({currentIndex + 1}/{images.length})</h1>
        <button className="btn" onClick={saveAll}><Save size={18} /> SAVE & TRAIN</button>
      </div>

      <div className="card" style={{display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative'}}>
        <div style={{position: 'relative', background: '#000', borderRadius: '8px', overflow: 'hidden'}}>
          <img 
            ref={imgRef}
            src={`${API_BASE}/data/runs/${runId}/images/${images[currentIndex]}`} 
            style={{display: 'block', maxWidth: '100%', maxHeight: '70vh'}}
            onLoad={(e) => {
              canvasRef.current.width = e.target.width;
              canvasRef.current.height = e.target.height;
              draw();
            }}
          />
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            style={{position: 'absolute', top: 0, left: 0, cursor: 'crosshair'}}
          />
        </div>

        <div style={{display: 'flex', gap: '2rem', marginTop: '1.5rem'}}>
          <button 
            className="btn btn-secondary" 
            disabled={currentIndex === 0}
            onClick={() => setCurrentIndex(c => c - 1)}
          >
            <ChevronLeft /> PREV
          </button>
          <button 
            className="btn btn-secondary" 
            disabled={currentIndex === images.length - 1}
            onClick={() => setCurrentIndex(c => c + 1)}
          >
            NEXT <ChevronRight />
          </button>
        </div>
      </div>
    </div>
  );
}
