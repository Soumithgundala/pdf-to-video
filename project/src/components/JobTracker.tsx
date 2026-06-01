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
  phase_1_status?: string;
  phase_1_progress?: number;
  phase_1_message?: string;
  phase_2_status?: string;
  phase_2_progress?: number;
  phase_2_message?: string;
  phase_3_status?: string;
  phase_3_progress?: number;
  phase_3_message?: string;
  phase_4_status?: string;
  phase_4_progress?: number;
  phase_4_message?: string;
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

  const phases = [
    {
      id: 'phase_1',
      title: 'Phase 1: PDF Processing & Panel Extraction',
      status: job?.phase_1_status || 'pending',
      progress: job?.phase_1_progress || 0,
      message: job?.phase_1_message,
      pendingMsg: 'Phase 1 is pending',
      completedMsg: 'Phase 1 is completed',
    },
    {
      id: 'phase_2',
      title: 'Phase 2: LLM Story Director',
      status: job?.phase_2_status || 'pending',
      progress: job?.phase_2_progress || 0,
      message: job?.phase_2_message,
      pendingMsg: 'Phase 2 is pending',
      completedMsg: 'Phase 2 is completed',
    },
    {
      id: 'phase_3',
      title: 'Phase 3: Audio Generation',
      status: job?.phase_3_status || 'pending',
      progress: job?.phase_3_progress || 0,
      message: job?.phase_3_message,
      pendingMsg: 'Phase 3 is pending',
      completedMsg: 'Phase 3 is completed',
    },
    {
      id: 'phase_4',
      title: 'Phase 4: Video Assembly',
      status: job?.phase_4_status || 'pending',
      progress: job?.phase_4_progress || 0,
      message: job?.phase_4_message,
      pendingMsg: 'Phase 4 is pending',
      completedMsg: 'Phase 4 is completed',
    },
  ];

  const getPhaseIcon = (phaseStatus: string) => {
    switch (phaseStatus) {
      case 'completed':
        return (
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-950/80 border border-emerald-500 text-emerald-400 z-10">
            <CheckCircle className="w-4 h-4" />
          </div>
        );
      case 'processing':
        return (
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-cyan-950/80 border border-cyan-500 text-cyan-400 animate-pulse z-10">
            <Loader2 className="w-4 h-4 animate-spin" />
          </div>
        );
      case 'failed':
        return (
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-red-950/80 border border-red-500 text-red-400 z-10">
            <XCircle className="w-4 h-4" />
          </div>
        );
      default:
        return (
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-slate-900 border border-slate-800 text-slate-500 z-10">
            <Clock className="w-4 h-4" />
          </div>
        );
    }
  };

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

      {/* Pipeline Stages Card */}
      {(status === 'processing' || status === 'completed' || status === 'failed') && (
        <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800 space-y-6">
          <h3 className="text-lg font-semibold text-white">Pipeline Execution Stages</h3>
          <div className="relative border-l border-slate-800 ml-4 pl-6 space-y-8 py-2">
            {phases.map((phase, idx) => {
              const isActive = phase.status === 'processing';
              const isDone = phase.status === 'completed';
              const isFailed = phase.status === 'failed';

              let displayMsg = '';
              if (isDone) {
                displayMsg = phase.completedMsg;
              } else if (isActive) {
                displayMsg = phase.message || 'Processing…';
              } else if (isFailed) {
                displayMsg = phase.message || 'Failed';
              } else {
                displayMsg = phase.pendingMsg;
              }

              return (
                <div key={phase.id} className="relative flex items-start gap-4">
                  {/* Left Connector Line */}
                  {idx < phases.length - 1 && (
                    <div
                      className={`absolute left-4 top-8 bottom-0 w-0.5 -ml-[1px] ${
                        isDone ? 'bg-emerald-500' : isActive ? 'bg-slate-800' : 'bg-slate-800'
                      }`}
                      style={{ height: 'calc(100% + 2rem)' }}
                    />
                  )}

                  {/* Icon */}
                  {getPhaseIcon(phase.status)}

                  {/* Content */}
                  <div className="flex-1 min-w-0 pt-0.5 space-y-1">
                    <h4 className={`text-sm font-semibold transition-colors duration-300 ${
                      isActive ? 'text-cyan-400' : isDone ? 'text-slate-200' : 'text-slate-400'
                    }`}>
                      {phase.title}
                    </h4>

                    <p className={`text-xs ${
                      isActive ? 'text-cyan-400 animate-pulse font-medium' : isDone ? 'text-emerald-400 font-medium' : isFailed ? 'text-red-400 font-medium' : 'text-slate-500'
                    }`}>
                      {displayMsg}
                    </p>

                    {/* Progress Bar (Only show if processing and has progress > 0) */}
                    {isActive && phase.progress > 0 && (
                      <div className="mt-2 max-w-md">
                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-300"
                            style={{ width: `${phase.progress * 100}%` }}
                          />
                        </div>
                        <span className="text-[10px] text-slate-400">
                          {Math.round(phase.progress * 100)}% complete
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

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
