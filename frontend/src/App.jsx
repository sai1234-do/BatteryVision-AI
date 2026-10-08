
import { useState } from "react";
import "./styles.css";

const API_BASE_URL =
  "https://batteryvision-ai.onrender.com";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [result, setResult] = useState(null);
  const [gradcam, setGradcam] = useState(null);

  const [analyzing, setAnalyzing] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState("");

  function handleFile(file) {
    if (!file) return;

    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/webp",
    ];

    if (!allowedTypes.includes(file.type)) {
      setError("Unsupported image format. Use JPG, PNG or WEBP.");
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));

    setResult(null);
    setGradcam(null);
    setError("");
  }

  function handleFileChange(event) {
    handleFile(event.target.files[0]);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);

    const file = event.dataTransfer.files[0];
    handleFile(file);
  }

  async function analyzeImage() {
    if (!selectedFile) {
      setError("Select an image before starting inspection.");
      return;
    }

    setAnalyzing(true);
    setError("");
    setResult(null);
    setGradcam(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        `${API_BASE_URL}/predict?explain=true`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("AI analysis failed.");
      }

      const data = await response.json();

      setResult(data);

      if (data.gradcam) {
        setGradcam(
          `data:image/png;base64,${data.gradcam}`
        );
      }
    } catch (err) {
      console.error(err);
      setError(
        "Unable to connect to the BatteryVision AI backend."
      );
    } finally {
      setAnalyzing(false);
    }
  }

  async function generateReport() {
    if (!selectedFile || !result) {
      setError("Analyze an image before generating the report.");
      return;
    }

    setGeneratingReport(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        `${API_BASE_URL}/report`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(
          `Report generation failed: ${response.status}`
        );
      }

      const blob = await response.blob();

      const downloadUrl =
        window.URL.createObjectURL(blob);

      const link = document.createElement("a");

      link.href = downloadUrl;

      link.download =
        `BatteryVision_Inspection_Report_${Date.now()}.docx`;

      document.body.appendChild(link);

      link.click();

      link.remove();
      window.URL.revokeObjectURL(downloadUrl);

    } catch (err) {
      console.error("Report generation error:", err);

      setError(
        "Unable to generate the inspection report."
      );
    } finally {
      setGeneratingReport(false);
    }
  }

  function resetInspection() {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setGradcam(null);
    setError("");
  }

  return (
    <div className="app">

      {/* HEADER */}
      <header className="topbar">

        <div className="brand">
          <div className="brand-mark">
            BV
          </div>

          <div>
            <div className="brand-name">
              BATTERY<span>VISION</span> AI
            </div>

            <div className="brand-subtitle">
              AI-ASSISTED SURFACE INSPECTION
            </div>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          SYSTEM READY
        </div>

      </header>


      {/* MAIN */}
      <main className="main">

        {/* HERO */}
        <section className="hero">

          <div>
            <div className="eyebrow">
              COMPUTER VISION / INSPECTION
            </div>

            <h1>
              Inspect.
              <br />
              <span>Understand.</span>
            </h1>

            <p>
              Upload a battery surface image and let
              BatteryVision AI analyze its visual
              characteristics using a fine-tuned ResNet18
              model.
            </p>
          </div>

          <div className="model-badge">
            <span>MODEL</span>
            <strong>RESNET18</strong>
            <small>FINE-TUNED</small>
          </div>

        </section>


        {/* UPLOAD / RESULT */}
        <section className="inspection-grid">

          {/* IMAGE PANEL */}
          <div className="panel image-panel">

            <div className="panel-header">
              <div>
                <span className="panel-index">
                  01
                </span>

                <span className="panel-title">
                  INSPECTION INPUT
                </span>
              </div>

              {selectedFile && (
                <button
                  className="text-button"
                  onClick={resetInspection}
                >
                  CLEAR
                </button>
              )}
            </div>


            {!previewUrl ? (

              <label
                className={`upload-zone ${
                  dragActive ? "drag-active" : ""
                }`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() =>
                  setDragActive(false)
                }
                onDrop={handleDrop}
              >

                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleFileChange}
                  hidden
                />

                <div className="upload-icon">
                  +
                </div>

                <h3>
                  DROP BATTERY IMAGE
                </h3>

                <p>
                  Drag and drop an inspection image here
                </p>

                <span className="upload-format">
                  JPG · PNG · WEBP
                </span>

                <span className="select-button">
                  SELECT IMAGE
                </span>

              </label>

            ) : (

              <div className="preview-container">

                <img
                  src={previewUrl}
                  alt="Selected battery"
                  className="preview-image"
                />

                <div className="image-meta">

                  <div>
                    <span>FILE</span>
                    <strong>
                      {selectedFile.name}
                    </strong>
                  </div>

                  <div>
                    <span>TYPE</span>
                    <strong>
                      {selectedFile.type}
                    </strong>
                  </div>

                  <div>
                    <span>SIZE</span>
                    <strong>
                      {(selectedFile.size / 1024).toFixed(1)}
                      {" "}KB
                    </strong>
                  </div>

                </div>

              </div>

            )}

          </div>


          {/* RESULT PANEL */}
          <div className="panel result-panel">

            <div className="panel-header">

              <div>
                <span className="panel-index">
                  02
                </span>

                <span className="panel-title">
                  AI INSPECTION
                </span>
              </div>

              {result && (
                <span className="live-label">
                  ANALYSIS COMPLETE
                </span>
              )}

            </div>


            {!result ? (

              <div className="empty-result">

                <div className="scan-line"></div>

                <div className="result-placeholder">
                  <span>
                    {analyzing
                      ? "ANALYZING IMAGE..."
                      : "AWAITING INSPECTION"}
                  </span>

                  <small>
                    {analyzing
                      ? "ResNet18 is processing the image"
                      : "Upload an image to begin"}
                  </small>
                </div>

              </div>

            ) : (

              <div className="result-content">

                <div className="result-label">
                  AI PREDICTION
                </div>

                <div className="prediction">
                  {result.prediction}
                </div>

                <div className="confidence-row">

                  <div>
                    <span className="confidence-number">
                      {result.confidence_percent.toFixed(2)}%
                    </span>

                    <span className="confidence-label">
                      MODEL CONFIDENCE
                    </span>
                  </div>

                  <div
                    className={`confidence-status ${
                      (
                        result.confidence_level ||
                        (
                          result.confidence_percent >= 80
                            ? "High"
                            : result.confidence_percent >= 60
                              ? "Moderate"
                              : "Low"
                        )
                      ).toLowerCase()
                    }`}
                  >
                    <span></span>

                    {result.confidence_level || (
                      result.confidence_percent >= 80
                        ? "High"
                        : result.confidence_percent >= 60
                          ? "Moderate"
                          : "Low"
                    )}
                  </div>

                  <div className="confidence-meter">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${result.confidence_percent}%`,
                      }}
                    ></div>
                  </div>

                </div>

                <div className="result-details">

                  <div>
                    <span>MODEL</span>
                    <strong>
                      {result.model}
                    </strong>
                  </div>

                  <div>
                    <span>EXPLANATION</span>
                    <strong>
                      Grad-CAM generated
                    </strong>
                  </div>

                </div>

              </div>

            )}

          </div>

        </section>


        {/* ANALYZE BUTTON */}
        <div className="action-row">

          <button
            className="analyze-button"
            onClick={analyzeImage}
            disabled={
              !selectedFile || analyzing
            }
          >

            <span>
              {analyzing
                ? "ANALYZING..."
                : "ANALYZE IMAGE"}
            </span>

            <span className="arrow">
              →
            </span>

          </button>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

        </div>


        {/* EVIDENCE */}
        {result && gradcam && (

          <section className="evidence-section">

            <div className="section-heading">

              <div>
                <span className="panel-index">
                  03
                </span>

                <span className="panel-title">
                  AI ATTENTION ANALYSIS
                </span>
              </div>

              <span className="section-note">
                GRAD-CAM / VISUAL EVIDENCE
              </span>

            </div>


            <div className="evidence-grid">

              <div className="evidence-image">

                <div className="evidence-label">
                  ORIGINAL IMAGE
                </div>

                <img
                  src={previewUrl}
                  alt="Original battery"
                />

              </div>


              <div className="evidence-image">

                <div className="evidence-label">
                  MODEL ATTENTION
                </div>

                <img
                  src={gradcam}
                  alt="Grad-CAM attention"
                />

              </div>

            </div>


            <div className="evidence-description">

              <div className="evidence-indicator"></div>

              <div>
                <strong>
                  Surface-focused attention detected.
                </strong>

                <p>
                  The model's prediction was associated
                  with localized regions of the battery
                  surface rather than obvious background
                  or border regions.
                </p>
              </div>

            </div>

            <div className="disclaimer">
              GRAD-CAM INDICATES REGIONS ASSOCIATED WITH
              THE MODEL PREDICTION. IT DOES NOT ESTABLISH
              CAUSAL REASONING.
            </div>

          </section>

        )}


        {/* PIPELINE */}
        <section className="pipeline-section">

          <div className="section-heading">

            <div>
              <span className="panel-index">
                04
              </span>

              <span className="panel-title">
                INSPECTION PIPELINE
              </span>
            </div>

          </div>


          <div className="pipeline">

            {[
              ["01", "INPUT"],
              ["02", "PREPROCESS"],
              ["03", "RESNET18"],
              ["04", "PREDICTION"],
              ["05", "EXPLAIN"],
            ].map(([number, name], index) => (

              <div
                className="pipeline-step"
                key={name}
              >

                <span className="pipeline-number">
                  {number}
                </span>

                <strong>
                  {name}
                </strong>

                {index < 4 && (
                  <span className="pipeline-arrow">
                    →
                  </span>
                )}

              </div>

            ))}

          </div>

        </section>


        {/* REPORT */}
        <section className="report-section">

          <div>

            <div className="eyebrow">
              DOCUMENTATION
            </div>

            <h2>
              Generate inspection evidence.
            </h2>

            <p>
              Create a detailed AI inspection report
              containing the prediction, confidence,
              original image, Grad-CAM evidence and
              model information.
            </p>

          </div>

          <button
            className="report-button"
            onClick={generateReport}
            disabled={!result || generatingReport}
          >
            {generatingReport
              ? "GENERATING REPORT..."
              : "GENERATE INSPECTION REPORT"
            }
            <span>↗</span>
          </button>

        </section>

      </main>


      {/* FOOTER */}
      <footer>

        <span>
          BATTERYVISION AI
        </span>

        <span>
          AI-ASSISTED INSPECTION · RESNET18
        </span>

        <span>
          DEVELOPMENT BUILD
        </span>

      </footer>

    </div>
  );
}

export default App;