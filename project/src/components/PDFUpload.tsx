import { useState, useCallback } from 'react';
import { Upload, FileText, AlertCircle, Loader2, Sliders, Sparkles, Zap, Image as ImageIcon, X, Plus } from 'lucide-react';
import { uploadPDF, processJob, uploadCharacterReferences } from '../lib/api';
import { useAuth } from '../hooks/useAuth';

interface PDFUploadProps {
  onUploadSuccess: (jobId: string) => void;
}

export function PDFUpload({ onUploadSuccess }: PDFUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [colorizerMode, setColorizerMode] = useState<string>('stable_diffusion');
  const [llmProvider, setLlmProvider] = useState<string>('google');
  const [characterFiles, setCharacterFiles] = useState<File[]>([]);
  const { user } = useAuth();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setError(null);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'application/pdf') {
      setFile(droppedFile);
    } else {
      setError('Please upload a PDF file');
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.type === 'application/pdf') {
        setFile(selectedFile);
      } else {
        setError('Please upload a PDF file');
      }
    }
  }, []);

  const handleUpload = async () => {
    if (!file || !user) return;

    setUploading(true);
    setError(null);

    try {
      const { job_id } = await uploadPDF(file);
      if (characterFiles.length > 0) {
        await uploadCharacterReferences(job_id, characterFiles);
      }
      await processJob(job_id, { colorizerMode, llmProvider });
      onUploadSuccess(job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="flex items-center gap-2 p-4 bg-red-950/50 border border-red-900 rounded-lg text-red-400">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <p className="text-sm">{error}</p>
        </div>
      )}

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-2xl p-12 transition-all ${
          isDragging
            ? 'border-cyan-500 bg-cyan-950/20'
            : 'border-slate-700 hover:border-slate-600 bg-slate-900'
        }`}
      >
        <input
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />

        <div className="flex flex-col items-center justify-center text-center">
          <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 transition-all ${
            isDragging ? 'bg-cyan-500/20' : 'bg-slate-800'
          }`}>
            <Upload className={`w-8 h-8 transition-colors ${
              isDragging ? 'text-cyan-400' : 'text-slate-400'
            }`} />
          </div>

          {file ? (
            <>
              <div className="flex items-center gap-2 text-cyan-400 mb-2">
                <FileText className="w-5 h-5" />
                <span className="font-medium">{file.name}</span>
              </div>
              <p className="text-sm text-slate-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </>
          ) : (
            <>
              <p className="text-lg font-medium text-white mb-2">
                Drop your manga PDF here
              </p>
              <p className="text-sm text-slate-400 mb-4">
                or click to browse
              </p>
              <p className="text-xs text-slate-500">
                Supports PDF files up to 50MB
              </p>
            </>
          )}
        </div>
      </div>

      {file && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-300">
          {/* Character Reference Upload Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
              <ImageIcon className="w-5 h-5 text-cyan-400" />
              <h3 className="font-semibold text-white">Character Reference Guides (Optional)</h3>
            </div>
            <p className="text-xs text-slate-400">
              Upload colored images of characters (e.g. Luffy, Zoro). The AI will examine their colors (skin, clothing, hair) and apply them to the black-and-white panels.
            </p>

            <div className="relative border-2 border-dashed border-slate-700 hover:border-slate-600 rounded-xl p-4 transition-all bg-slate-950/50 flex flex-col items-center justify-center text-center cursor-pointer">
              <input
                type="file"
                multiple
                accept="image/*"
                onChange={(e) => {
                  const selected = Array.from(e.target.files || []);
                  setCharacterFiles(prev => [...prev, ...selected]);
                }}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <Plus className="w-6 h-6 text-slate-500 mb-1" />
              <span className="text-sm font-medium text-slate-300">Choose Images</span>
              <span className="text-xs text-slate-500 mt-0.5">PNG, JPG, WEBP</span>
            </div>

            {characterFiles.length > 0 && (
              <div className="space-y-2 mt-2">
                <div className="text-xs font-semibold text-slate-400">Selected Guides:</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-40 overflow-y-auto pr-1">
                  {characterFiles.map((f, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 bg-slate-950 border border-slate-800 rounded-lg text-xs">
                      <div className="flex items-center gap-2 text-slate-300 truncate">
                        <ImageIcon className="w-4 h-4 text-cyan-500 flex-shrink-0" />
                        <span className="truncate">{f.name}</span>
                        <span className="text-[10px] text-slate-500 flex-shrink-0">
                          ({(f.size / 1024).toFixed(0)} KB)
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => setCharacterFiles(prev => prev.filter((_, i) => i !== idx))}
                        className="text-slate-500 hover:text-red-400 transition-colors p-1"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Generation Settings */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
              <Sliders className="w-5 h-5 text-cyan-400" />
              <h3 className="font-semibold text-white">Generation Settings</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Colorizer Mode Selector */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">
                  Colorization Mode
                </label>
                <div className="grid grid-cols-1 gap-2">
                  <button
                    type="button"
                    onClick={() => setColorizerMode('stable_diffusion')}
                    className={`flex items-start gap-3 p-3 rounded-xl border text-left transition-all ${
                      colorizerMode === 'stable_diffusion'
                        ? 'border-cyan-500 bg-cyan-950/20 text-cyan-200'
                        : 'border-slate-800 hover:border-slate-700 bg-slate-950 text-slate-400'
                    }`}
                  >
                    <Sparkles className={`w-5 h-5 mt-0.5 flex-shrink-0 ${
                      colorizerMode === 'stable_diffusion' ? 'text-cyan-400' : 'text-slate-500'
                    }`} />
                    <div>
                      <div className="text-sm font-medium text-white">Stable Diffusion AI</div>
                      <div className="text-xs text-slate-500 mt-0.5">High-quality, rich character colors. Slow (20-30 mins).</div>
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() => setColorizerMode('classic')}
                    className={`flex items-start gap-3 p-3 rounded-xl border text-left transition-all ${
                      colorizerMode === 'classic'
                        ? 'border-blue-500 bg-blue-950/20 text-blue-200'
                        : 'border-slate-800 hover:border-slate-700 bg-slate-950 text-slate-400'
                    }`}
                  >
                    <Zap className={`w-5 h-5 mt-0.5 flex-shrink-0 ${
                      colorizerMode === 'classic' ? 'text-blue-400' : 'text-slate-500'
                    }`} />
                    <div>
                      <div className="text-sm font-medium text-white">Classic CNN (SIGGRAPH 17)</div>
                      <div className="text-xs text-slate-500 mt-0.5">Standard colors. Extremely fast (takes ~30s).</div>
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() => setColorizerMode('none')}
                    className={`flex items-start gap-3 p-3 rounded-xl border text-left transition-all ${
                      colorizerMode === 'none'
                        ? 'border-amber-500 bg-amber-950/20 text-amber-200'
                        : 'border-slate-800 hover:border-slate-700 bg-slate-950 text-slate-400'
                    }`}
                  >
                    <FileText className={`w-5 h-5 mt-0.5 flex-shrink-0 ${
                      colorizerMode === 'none' ? 'text-amber-400' : 'text-slate-500'
                    }`} />
                    <div>
                      <div className="text-sm font-medium text-white">Original Black & White</div>
                      <div className="text-xs text-slate-500 mt-0.5">Instant speed. Keeps classic manga style. (0s).</div>
                    </div>
                  </button>
                </div>
              </div>

              {/* LLM Provider Selector */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">
                  LLM Story Director
                </label>
                <div className="grid grid-cols-1 gap-2">
                  <button
                    type="button"
                    onClick={() => setLlmProvider('google')}
                    className={`flex items-start gap-3 p-3 rounded-xl border text-left transition-all ${
                      llmProvider === 'google'
                        ? 'border-cyan-500 bg-cyan-950/20 text-cyan-200'
                        : 'border-slate-800 hover:border-slate-700 bg-slate-950 text-slate-400'
                    }`}
                  >
                    <div className="w-5 h-5 flex items-center justify-center font-bold text-xs bg-slate-800 rounded-full text-cyan-400 flex-shrink-0">
                      G
                    </div>
                    <div>
                      <div className="text-sm font-medium text-white">(Recommended) Google Gemini</div>
                      <div className="text-xs text-slate-500 mt-0.5">Excellent multimodal parsing & timing.</div>
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() => setLlmProvider('openai')}
                    className={`flex items-start gap-3 p-3 rounded-xl border text-left transition-all ${
                      llmProvider === 'openai'
                        ? 'border-emerald-500 bg-emerald-950/20 text-emerald-200'
                        : 'border-slate-800 hover:border-slate-700 bg-slate-950 text-slate-400'
                    }`}
                  >
                    <div className="w-5 h-5 flex items-center justify-center font-bold text-xs bg-slate-800 rounded-full text-emerald-400 flex-shrink-0">
                      O
                    </div>
                    <div>
                      <div className="text-sm font-medium text-white">OpenAI GPT-4o</div>
                      <div className="text-xs text-slate-500 mt-0.5">Standard story narration and scripting.</div>
                    </div>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {file && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          className="w-full py-4 px-6 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold rounded-xl hover:from-cyan-600 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-slate-950 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {uploading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Upload className="w-5 h-5" />
              Start Processing
            </>
          )}
        </button>
      )}
    </div>
  );
}
