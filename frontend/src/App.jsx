import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, NavLink } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import AnnotationPage from './pages/AnnotationPage';
import ResultsPage from './pages/ResultsPage';

function App() {
  return (
    <Router>
      <nav className="navbar">
        <div style={{fontWeight: 800, fontSize: '1.2rem', color: 'var(--accent-primary)'}}>
          VISION FORGE
        </div>
        <div className="nav-links">
          <NavLink to="/" end>UPLOAD</NavLink>
          <NavLink to="/annotate">ANNOTATE</NavLink>
          <NavLink to="/results">RESULTS</NavLink>
        </div>
      </nav>

      <main className="app-container">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/annotate/:runId" element={<AnnotationPage />} />
          <Route path="/annotate" element={<AnnotationPage />} />
          <Route path="/results/:runId" element={<ResultsPage />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
