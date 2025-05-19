import React, { useState } from 'react'; // Removed unused useEffect import
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import axios from 'axios';
import './App.css';
import Chatbot from './Chatbot';

// Tumor Analysis Component
const TumorAnalysis = () => {
  const [file, setFile] = useState(null);
  const [niftiFile, setNiftiFile] = useState(null);
  const [result, setResult] = useState(null);
  const [report, setReport] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [segmentationImage, setSegmentationImage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [requiresNifti, setRequiresNifti] = useState(false);
  const [initialResult, setInitialResult] = useState(null);
  const [manualFeatures, setManualFeatures] = useState({
    t1_3d_tumor_volume: '',
    t1_3d_max_intensity: '',
    t1_3d_major_axis_length: '',
    t1_3d_area: '',
    t1_3d_minor_axis_length: '',
    t1_3d_extent: '',
    t1_3d_surface_to_volume_ratio: '',
    t1_3d_glcm_contrast: '',
    t1_3d_mean_intensity: '',
    t1_2d_area_median: ''
  });

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setReport('');
    setAudioUrl('');
    setSegmentationImage('');
    setError('');
    setRequiresNifti(false);
    setNiftiFile(null);
    setInitialResult(null);
    setManualFeatures({
      t1_3d_tumor_volume: '',
      t1_3d_max_intensity: '',
      t1_3d_major_axis_length: '',
      t1_3d_area: '',
      t1_3d_minor_axis_length: '',
      t1_3d_extent: '',
      t1_3d_surface_to_volume_ratio: '',
      t1_3d_glcm_contrast: '',
      t1_3d_mean_intensity: '',
      t1_2d_area_median: ''
    });
  };

  const handleNiftiFileChange = (e) => {
    setNiftiFile(e.target.files[0]);
    setError('');
  };

  const handleFeatureChange = (e) => {
    const { name, value } = e.target;
    setManualFeatures((prev) => ({ ...prev, [name]: value }));
  };

  const handleManualFeatureSubmission = async () => {
    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      Object.entries(manualFeatures).forEach(([key, value]) => {
        formData.append(key, value);
      });
      formData.append('tumor_type', result?.tumor_type || 'Non-Glioma');

      const response = await axios.post('http://localhost:5000/submit_manual_features', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setResult({
        ...result,
        survival_days: response.data.survival_days,
        features: response.data.features
      });
      setReport(response.data.report);
      setAudioUrl(response.data.audio_url);
    } catch (err) {
      setError(err.response?.data?.error || 'Error submitting manual features');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!file && !requiresNifti) {
      setError('Please select an initial file to analyze.');
      return;
    }

    if (requiresNifti && !niftiFile) {
      setError('Please select a 3D NIfTI file to continue.');
      return;
    }

    setLoading(true);
    setError('');
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    }
    if (niftiFile) {
      formData.append('nifti_file', niftiFile);
    }

    try {
      const response = await axios.post('http://localhost:5000/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (response.data.requires_nifti) {
        setRequiresNifti(true);
        setInitialResult({
          tumor_detected: response.data.tumor_detected,
          tumor_type: response.data.tumor_type
        });
      } else {
        setResult({
          tumor_detected: response.data.tumor_detected,
          tumor_type: response.data.tumor_type,
          survival_days: response.data.survival_days,
          features: response.data.features
        });
        setReport(response.data.report);
        setAudioUrl(response.data.audio_url);
        setSegmentationImage(response.data.segmentation_image);
        setRequiresNifti(false);
        setNiftiFile(null);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error analyzing the image. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handlePlayAudio = () => {
    if (audioUrl) {
      const audio = new Audio(`http://localhost:5000${audioUrl}`);
      audio.play().catch((err) => {
        setError('Error playing audio: ' + err.message);
        console.error(err);
      });
    } else {
      setError('No audio URL available. Please ensure analysis completed successfully.');
    }
  };

  const featureLabels = {
    t1_3d_tumor_volume: 'Tumor Volume (3D)',
    t1_3d_max_intensity: 'Max Intensity (3D)',
    t1_3d_major_axis_length: 'Major Axis Length (3D)',
    t1_3d_area: 'Area (3D)',
    t1_3d_minor_axis_length: 'Minor Axis Length (3D)',
    t1_3d_extent: 'Extent (3D)',
    t1_3d_surface_to_volume_ratio: 'Surface to Volume Ratio (3D)',
    t1_3d_glcm_contrast: 'GLCM Contrast (3D)',
    t1_3d_mean_intensity: 'Mean Intensity (3D)',
    t1_2d_area_median: 'Area Median (2D)'
  };

  return (
    <div className="app">
      {/* Decorative Sidebar */}
      <div className="sidebar">
        <span className="sidebar-icon">🧠</span>
        <span className="sidebar-icon">🩺</span>
        <span className="sidebar-icon">📋</span>
      </div>

      {/* Floating Info Bubble */}
      <div className="info-bubble">
        <p>Welcome! Upload an image to analyze tumors. 🤗</p>
      </div>

      <div className="container">
        <header className="header">
          <h1>Tumor Analysis and Reporting System</h1>
          <img
            src="https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg"
            alt="React Logo"
            className="logo"
          />
        </header>

        <div className="upload-section">
          <h2>Choose an Image</h2>
          <div className="file-input-wrapper">
            <input
              type="file"
              id="file-input"
              accept=".png,.nii,.nii.gz"
              onChange={handleFileChange}
              className="file-input"
              disabled={requiresNifti}
            />
            <label htmlFor="file-input" className="file-input-label">
              Choose a File - {file ? file.name : "No File Selected"}
            </label>
          </div>
          {requiresNifti && (
            <>
              <p style={{ color: '#e57373', margin: '10px 0' }}>
                Glioma detected. Please provide a 3D NIfTI file for segmentation.
              </p>
              <div className="file-input-wrapper">
                <input
                  type="file"
                  id="nifti-file-input"
                  accept=".nii,.nii.gz"
                  onChange={handleNiftiFileChange}
                  className="file-input"
                />
                <label htmlFor="nifti-file-input" className="file-input-label">
                  Choose a File - {niftiFile ? niftiFile.name : "No File Selected"}
                </label>
              </div>
            </>
          )}
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="analyze-button"
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        {error && <p className="error">{error}</p>}

        {(result || initialResult) && (
          <div className="results-section">
            <h2>Results</h2>
            <div className="result-item">
              <img
                src="https://cdn-icons-png.flaticon.com/512/1160/1160119.png"
                alt="Brain Icon"
                className="brain-icon"
              />
              <div>
                <p><strong>Tumor</strong></p>
                <p>{(result || initialResult).tumor_detected ? 'Detected' : 'Not Detected'}</p>
              </div>
            </div>
            {(result || initialResult).tumor_detected && (
              <>
                <p><strong>Diagnosis:</strong> {(result || initialResult).tumor_type}</p>
                {result && result.tumor_type !== 'Glioma' && (
                  <>
                    <p>Enter features manually to predict survival days:</p>
                    <div className="manual-features-section">
                      {Object.keys(manualFeatures).map((key) => (
                        <div key={key} className="feature-input">
                          <label>{featureLabels[key]}:</label>
                          <input
                            type="number"
                            name={key}
                            value={manualFeatures[key]}
                            onChange={handleFeatureChange}
                            placeholder="Enter value"
                            step="any"
                          />
                        </div>
                      ))}
                      <button
                        onClick={handleManualFeatureSubmission}
                        disabled={loading || Object.values(manualFeatures).some((val) => val === '')}
                        className="analyze-button"
                      >
                        {loading ? 'Submitting...' : 'Submit Features'}
                      </button>
                    </div>
                  </>
                )}
                {segmentationImage && result && result.tumor_type === 'Glioma' && (
                  <div className="segmentation-image">
                    <h3>Segmentation Result</h3>
                    <img
                      src={`http://localhost:5000${segmentationImage}`}
                      alt="Segmentation Result"
                    />
                    {result.features && (
                      <>
                        <p><strong>📏 Features extracted from segmentation mask:</strong></p>
                        <ul>
                          {Object.entries(result.features).map(([key, value]) => (
                            <li key={key}>
                              {key}: {value.toFixed(4)}
                            </li>
                          ))}
                        </ul>
                        {result.survival_days !== null && (
                          <p><strong>📅 Predicted Survival Days:</strong> {result.survival_days}</p>
                        )}
                      </>
                    )}
                  </div>
                )}
                {result && result.tumor_type !== 'Glioma' && result.features && result.survival_days !== null && (
                  <div className="manual-features-result">
                    <p><strong>📏 Manually Entered Features:</strong></p>
                    <ul>
                      {Object.entries(result.features).map(([key, value]) => (
                        <li key={key}>
                          {key}: {value.toFixed(4)}
                        </li>
                      ))}
                    </ul>
                    <p><strong>📅 Predicted Survival Days:</strong> {result.survival_days}</p>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {report && (
          <div className="report-section">
            <h2>Patient Report</h2>
            <p>{report}</p>
            <button onClick={handlePlayAudio} className="play-button">
              <span className="play-icon">▶</span> Play Report
            </button>
          </div>
        )}

        <footer className="footer">
          <p>Made with 💖 by Fatma Hammedi</p>
        </footer>
      </div>

      {/* Chatbot Icon - Moved Outside Container */}
      <Link
        to="/chatbot"
        className="chatbot-icon"
        onClick={() => console.log("Chatbot icon clicked, navigating to /chatbot")}
      >
        <img
          src="https://cdn-icons-png.flaticon.com/512/8943/8943377.png"
          alt="Chatbot Icon"
        />
      </Link>

      {/* Wave Animation */}
      <div className="wave-container">
        <div className="wave"></div>
        <div className="wave"></div>
      </div>
    </div>
  );
};

// Main App Component with Routing
const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<TumorAnalysis />} />
        <Route path="/chatbot" element={<Chatbot />} />
      </Routes>
    </Router>
  );
};

export default App;