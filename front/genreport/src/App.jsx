import { useMemo, useState } from "react"
import axios from "axios"
import { Upload, FileImage, Loader, AlertCircle, CheckCircle } from "lucide-react"
import "./App.css"

/**
 * @typedef {Object} Metrics
 * @property {number} top_k
 * @property {number} max_distance
 * @property {number} dedup_similarity_threshold
 * @property {number} cluster_distance_threshold
 * @property {number} retrieved_count
 * @property {number} filtered_count
 * @property {number} deduped_count
 * @property {number} clusters_count
 * @property {number} compact_count
 * @property {number | null} alignment_cosine
 * @property {number | null} alignment_percent
 * @property {string=} note
 */

/**
 * @typedef {Object} AnalyzeResponse
 * @property {string} report
 * @property {string} ranked_findings
 * @property {string[]} compact_findings
 * @property {Metrics} metrics
 */

const api = axios.create({
  baseURL: "http://localhost:8000", // FastAPI base URL
  timeout: 120000, // 2 minutes (LLM + retrieval can take time)
})

export default function XrayAnalyzer() {
  /** @type {[File|null, Function]} */
  const [image, setImage] = useState(null)
  /** @type {[string|null, Function]} */
  const [imagePreview, setImagePreview] = useState(null)

  /** @type {[AnalyzeResponse|null, Function]} */
  const [result, setResult] = useState(null)

  /** @type {[boolean, Function]} */
  const [loading, setLoading] = useState(false)
  /** @type {[string|null, Function]} */
  const [error, setError] = useState(null)

  const prettyAlignment = useMemo(() => {
    const v = result?.metrics?.alignment_percent
    if (v === null || v === undefined) return "—"
    // keep 2 decimals
    return `${v.toFixed(2)}%`
  }, [result])

  const handleImageUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setImage(file)
    setError(null)
    setResult(null)

    const reader = new FileReader()
    reader.onloadend = () => setImagePreview(reader.result)
    reader.readAsDataURL(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()

    const file = e.dataTransfer.files?.[0]
    if (!file) return

    setImage(file)
    setError(null)
    setResult(null)

    const reader = new FileReader()
    reader.onloadend = () => setImagePreview(reader.result)
    reader.readAsDataURL(file)
  }

  const humanAxiosError = (err) => {
    // Axios error shapes vary; we keep it safe
    const status = err?.response?.status
    const detail = err?.response?.data?.detail
    const msg = err?.message

    if (detail) return `Server error${status ? ` (${status})` : ""}: ${detail}`
    if (status) return `Request failed (${status}).`
    return msg || "Failed to analyze image. Please try again."
  }

  const handleAnalyze = async () => {
    if (!image) {
      setError("Please upload an image first")
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()

      // IMPORTANT: FastAPI script expects key "file" by default:
      // async def predict(file: UploadFile = File(...))
      formData.append("file", image)

      /** @type {{data: AnalyzeResponse}} */
      const response = await api.post("/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })

      setResult(response.data)
    } catch (err) {
      setError(humanAxiosError(err))
      console.error("Analyze error:", err)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setImage(null)
    setImagePreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-blue-100 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-blue-100 to-blue-50 rounded-xl border border-blue-200">
              <FileImage className="w-7 h-7 text-blue-600" strokeWidth={1.5} />
            </div>
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-600 to-blue-700 bg-clip-text text-transparent">
                X-Ray Analysis
              </h1>
              <p className="text-sm text-gray-600 mt-1">Chest X-Ray Report Generator</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Upload & Preview */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg border border-blue-100 p-8 hover:shadow-xl transition-shadow duration-300">
              <h2 className="text-2xl font-bold text-gray-900 mb-8 flex items-center gap-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Upload className="w-6 h-6 text-blue-600" strokeWidth={1.5} />
                </div>
                Upload X-Ray Image
              </h2>

              {/* Upload Area */}
              {!imagePreview && (
                <div
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  className="border-2 border-dashed border-blue-300 rounded-xl p-12 text-center hover:border-blue-500 hover:bg-blue-50/50 transition-all duration-300 cursor-pointer group"
                >
                  <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" id="image-input" />
                  <label htmlFor="image-input" className="cursor-pointer block">
                    <Upload className="w-16 h-16 text-blue-300 mx-auto mb-4 group-hover:text-blue-400 transition-colors" strokeWidth={1.5} />
                    <p className="text-gray-800 font-semibold text-lg">Drag and drop your image here</p>
                    <p className="text-gray-500 text-sm mt-2">or click to browse your files</p>
                    <p className="text-gray-400 text-xs mt-3">Supported formats: JPG, PNG, WEBP</p>
                  </label>
                </div>
              )}

              {/* Image Preview */}
              {imagePreview && (
                <div className="mt-8 space-y-4">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold text-gray-700">Preview</p>
                    <span className="text-xs bg-blue-100 text-blue-700 px-3 py-1 rounded-full font-medium">
                      {(image.size / 1024).toFixed(2)} KB
                    </span>
                  </div>

                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl overflow-hidden border border-gray-200 shadow-md">
                    <img
                      src={imagePreview || "/placeholder.svg"}
                      alt="X-Ray Preview"
                      className="w-full h-auto max-h-96 object-contain p-4"
                    />
                  </div>

                  <p className="text-xs text-gray-600 truncate">📄 {image.name}</p>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="mt-8 p-4 bg-red-50 border border-red-200 rounded-xl flex gap-3 animate-slideIn">
                  <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" strokeWidth={2} />
                  <p className="text-sm text-red-700 font-medium">{error}</p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="mt-8 flex gap-3 flex-col sm:flex-row">
                <button
                  onClick={handleAnalyze}
                  disabled={!imagePreview || loading}
                  className="flex-1 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:from-gray-300 disabled:to-gray-400 text-white font-bold py-3 px-6 rounded-xl transition-all duration-300 flex items-center justify-center gap-2 shadow-lg hover:shadow-xl disabled:shadow-none disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <Loader className="w-5 h-5 animate-spin" strokeWidth={2} />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Upload className="w-5 h-5" strokeWidth={2} />
                      <span>Analyze Image</span>
                    </>
                  )}
                </button>

                {imagePreview && (
                  <button
                    onClick={handleReset}
                    disabled={loading}
                    className="bg-gray-100 hover:bg-gray-200 disabled:bg-gray-100 text-gray-700 font-semibold py-3 px-6 rounded-xl transition-all duration-300 disabled:cursor-not-allowed"
                  >
                    Reset
                  </button>
                )}
              </div>
            </div>

            {/* Metrics Card (only show after result) */}
            {result?.metrics && (
              <div className="bg-white rounded-2xl shadow-lg border border-blue-100 p-8">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Metrics</h3>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <MetricRow label="Alignment" value={prettyAlignment} />
                  <MetricRow label="Top K" value={String(result.metrics.top_k)} />
                  <MetricRow label="Retrieved" value={String(result.metrics.retrieved_count)} />
                  <MetricRow label="Filtered" value={String(result.metrics.filtered_count)} />
                  <MetricRow label="Deduped" value={String(result.metrics.deduped_count)} />
                  <MetricRow label="Clusters" value={String(result.metrics.clusters_count)} />
                  <MetricRow label="Compact findings" value={String(result.metrics.compact_count)} />
                </div>

                {result.metrics.note && (
                  <p className="text-sm text-gray-600 mt-4">{result.metrics.note}</p>
                )}
              </div>
            )}
          </div>

          {/* Right Column - Report */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg border border-blue-100 p-8 min-h-96 hover:shadow-xl transition-shadow duration-300">
              <h2 className="text-2xl font-bold text-gray-900 mb-8 flex items-center gap-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <FileImage className="w-6 h-6 text-blue-600" strokeWidth={1.5} />
                </div>
                Analysis Report
              </h2>

              {!result?.report && !loading && (
                <div className="flex flex-col items-center justify-center h-64 text-center">
                  <div className="p-4 bg-gray-100 rounded-full mb-4">
                    <FileImage className="w-12 h-12 text-gray-400" strokeWidth={1.5} />
                  </div>
                  <p className="text-gray-600 font-medium text-lg">No analysis yet</p>
                  <p className="text-gray-500 text-sm mt-2">Upload and analyze an X-ray image to see the report</p>
                </div>
              )}

              {loading && (
                <div className="flex flex-col items-center justify-center h-64">
                  <div className="relative w-16 h-16 mb-6">
                    <Loader className="w-16 h-16 text-blue-600 animate-spin" strokeWidth={1.5} />
                  </div>
                  <p className="text-gray-700 font-semibold text-lg">Analyzing your X-ray...</p>
                  <p className="text-gray-500 text-sm mt-3">This may take a few moments</p>
                </div>
              )}

              {result?.report && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
                    <CheckCircle className="w-6 h-6 text-green-600 flex-shrink-0" strokeWidth={2} />
                    <span className="font-semibold text-green-700">Analysis Complete</span>
                  </div>

                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6 border border-gray-200 max-h-96 overflow-y-auto shadow-inner">
                    <div className="text-gray-700 text-sm leading-relaxed whitespace-pre-wrap font-medium">
                      {result.report}
                    </div>
                  </div>

                  {/* Optional: show the ranked/compact findings used for generation */}
                  {result.ranked_findings && (
                    <details className="bg-white border border-gray-200 rounded-xl p-4">
                      <summary className="cursor-pointer text-sm font-semibold text-gray-700">
                        Show extracted findings used to generate the report
                      </summary>
                      <pre className="mt-3 text-xs whitespace-pre-wrap text-gray-700">
                        {result.ranked_findings}
                      </pre>
                    </details>
                  )}

                  <button
                    onClick={handleReset}
                    className="w-full bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-3 px-4 rounded-xl transition-all duration-300"
                  >
                    Analyze Another Image
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

function MetricRow({ label, value }) {
  return (
    <div className="flex items-center justify-between bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
      <span className="text-sm font-semibold text-gray-700">{label}</span>
      <span className="text-sm font-bold text-gray-900">{value}</span>
    </div>
  )
}
