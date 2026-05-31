import { useState, useEffect, useCallback } from 'react';
import { getVideoUrl, getJobStatus } from '../lib/api';
import {
  Loader2,
  CheckCircle,
  XCircle,
  Film,
  Download,
  RefreshCw,
  Clock,
  Image,
  FileVideo,
} from 'lucide-react';

interface JobTrackerProps {
  jobId: string;
  onReset: () => void;
}

interface JobState {
  status: string;
  progress: number;
  message?: string;
  pdf_filename?: string;
  total_pages?: number;
  total_panels?: number;
  error_message?: string;
}

export function JobTracker({ jobId, onReset }: JobTrackerProps) {
  const [job, setJob] = useState<JobState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const data = await getJobStatus(jobId);
      setJob((prev) => ({ ...prev, ...data }));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load job status');
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    if (job?.status === 'completed' || job?.status === 'failed') {
      return;
    }

    const interval = setInterval(() => {
      fetchStatus();
    }, 3000);

    return () => clearInterval(interval);
  }, [fetchStatus, job?.status]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-5 h-5 text-slate-400" />;
      case 'processing':
        return <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-emerald-400" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'text-slate-400';
      case 'processing':
        return 'text-cyan-400';
      case 'completed':
        return 'text-emerald-400';
      case 'failed':
        return 'text-red-400';
      default:
        return 'text-cyan-400';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'processing':
        return 'Processing…';
      case 'completed':
        return 'Completed';
      case 'failed':
        return 'Failed';
      default:
        return 'Pending…';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
      </div>
    );
  }

  if (error && !job) {
    return (
      <div className="text-center py-12">
        <XCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
        <p className="text-red-400">{error}</p>
        <button
          onClick={fetchStatus}
          className="mt-4 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors text-sm"
        >
          Retry
        </button>
      </div>
    );
  }

  const status = job?.status ?? 'processing';
  const progress = (job?.progress ?? 0) * 100;

  return (
    <div className="space-y-8">
      {/* Job Overview */}
      <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-white mb-1">
              {job?.pdf_filename ?? `Job ${jobId.slice(0, 8)}…`}
            </h2>
            {(job?.total_pages || job?.total_panels) && (
              <div className="flex items-center gap-4 text-sm text-slate-400">
                {job.total_pages && (
                  <span className="flex items-center gap-1">
                    <Image className="w-4 h-4" />
                    {job.total_pages} pages
                  </span>
                )}
                {job.total_panels && (
                  <span className="flex items-center gap-1">
                    <Film className="w-4 h-4" />
                    {job.total_panels} panels
                  </span>
                )}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            {getStatusIcon(status)}
            <span className={`font-medium capitalize ${getStatusColor(status)}`}>
              {getStatusLabel(status)}
            </span>
          </div>
        </div>

        {/* Progress bar */}
        {status === 'processing' && (
          <div className="mt-2">
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full transition-all duration-500"
                style={{ width: `${Math.max(progress, 5)}%` }}
              />
            </div>
            <p className="text-xs text-cyan-400 font-medium mt-1.5 animate-pulse">
              {job?.message || 'Pipeline running… this may take several minutes'}
            </p>
          </div>
        )}

        {status === 'failed' && job?.error_message && (
          <div className="mt-4 p-4 bg-red-950/50 border border-red-900 rounded-lg">
            <p className="text-red-400 text-sm">{job.error_message}</p>
          </div>
        )}
      </div>

      {/* Video players — shown when completed */}
      {status === 'completed' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-white">Video Parts</h3>
          {[1, 2, 3, 4].map((part) => (
            <div key={part} className="bg-slate-900 rounded-xl p-5 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-slate-800 flex items-center justify-center">
                    <span className="text-lg font-bold text-cyan-400">{part}</span>
                  </div>
                  <p className="font-medium text-white">Part {part}</p>
                </div>
                <a
                  href={getVideoUrl(jobId, part)}
                  download
                  className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg transition-colors text-sm font-medium"
                >
                  <Download className="w-4 h-4" />
                  Download
                </a>
              </div>
              {/* Inline video player */}
              <video
                key={getVideoUrl(jobId, part)}
                controls
                preload="metadata"
                className="w-full rounded-lg bg-black"
                style={{ maxHeight: '480px' }}
              >
                <source src={getVideoUrl(jobId, part)} type="video/mp4" />
                Your browser does not support video playback.
              </video>
            </div>
          ))}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-4">
        <button
          onClick={onReset}
          className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-colors font-medium"
        >
          <RefreshCw className="w-5 h-5" />
          Process Another
        </button>
      </div>
    </div>
  );
}
