import { useState } from 'react';
import { PDFUpload } from './PDFUpload';
import { JobTracker } from './JobTracker';
import { Header } from './Header';
import { Film, Sparkles, Zap, Wand2 } from 'lucide-react';

export function Dashboard() {
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);

  const handleUploadSuccess = (jobId: string) => {
    setCurrentJobId(jobId);
  };

  const handleReset = () => {
    setCurrentJobId(null);
  };

  return (
    <div className="min-h-screen bg-slate-950">
      <Header />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!currentJobId ? (
          <>
            {/* Hero Section */}
            <div className="text-center mb-12">
              <h1 className="text-4xl sm:text-5xl font-bold text-white mb-4">
                Turn Manga Chapters Into
                <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                  {' '}Video Recaps
                </span>
              </h1>
              <p className="text-lg text-slate-400 max-w-2xl mx-auto">
                Upload a manga PDF and watch AI transform it into a 4-part video miniseries
                with professional voiceover and cinematic editing.
              </p>
            </div>

            {/* Features */}
            <div className="grid sm:grid-cols-3 gap-4 mb-12">
              <div className="bg-slate-900 rounded-xl p-5 border border-slate-800">
                <div className="w-10 h-10 rounded-lg bg-cyan-950/50 flex items-center justify-center mb-3">
                  <Film className="w-5 h-5 text-cyan-400" />
                </div>
                <h3 className="font-semibold text-white mb-1">Panel Extraction</h3>
                <p className="text-sm text-slate-400">
                  AI automatically detects and extracts every panel from your manga pages.
                </p>
              </div>

              <div className="bg-slate-900 rounded-xl p-5 border border-slate-800">
                <div className="w-10 h-10 rounded-lg bg-cyan-950/50 flex items-center justify-center mb-3">
                  <Wand2 className="w-5 h-5 text-cyan-400" />
                </div>
                <h3 className="font-semibold text-white mb-1">Story Writing</h3>
                <p className="text-sm text-slate-400">
                  AI analyzes the narrative and creates engaging scripts for each part.
                </p>
              </div>

              <div className="bg-slate-900 rounded-xl p-5 border border-slate-800">
                <div className="w-10 h-10 rounded-lg bg-cyan-950/50 flex items-center justify-center mb-3">
                  <Zap className="w-5 h-5 text-cyan-400" />
                </div>
                <h3 className="font-semibold text-white mb-1">Pro Video Output</h3>
                <p className="text-sm text-slate-400">
                  Get polished 1080x1920 videos with voiceover and cinematic effects.
                </p>
              </div>
            </div>

            {/* Upload Section */}
            <div className="bg-slate-900 rounded-2xl p-6 sm:p-8 border border-slate-800">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">Upload Your Manga</h2>
                  <p className="text-sm text-slate-400">Start by uploading a PDF chapter</p>
                </div>
              </div>

              <PDFUpload onUploadSuccess={handleUploadSuccess} />
            </div>
          </>
        ) : (
          <JobTracker jobId={currentJobId} onReset={handleReset} />
        )}
      </main>
    </div>
  );
}
