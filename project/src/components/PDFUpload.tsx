import { useState, useCallback } from 'react';
import { Upload, FileText, AlertCircle, Loader2 } from 'lucide-react';
import { uploadPDF, processJob } from '../lib/api';
import { useAuth } from '../hooks/useAuth';

interface PDFUploadProps {
  onUploadSuccess: (jobId: string) => void;
}

export function PDFUpload({ onUploadSuccess }: PDFUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
      await processJob(job_id);
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
