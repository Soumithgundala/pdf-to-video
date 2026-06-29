import { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, Play, Square, Loader2, User, Users } from 'lucide-react';
import { getVoices, getVoiceSampleUrl, type Voice } from '../lib/api';

interface VoiceSelectorProps {
  selectedVoice: string;
  onVoiceChange: (voiceId: string) => void;
}

export function VoiceSelector({ selectedVoice, onVoiceChange }: VoiceSelectorProps) {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [playingVoice, setPlayingVoice] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'male' | 'female'>('male');
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getVoices()
      .then((v) => {
        if (!cancelled) {
          setVoices(v);
          setLoading(false);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e.message);
          setLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, []);

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
    }
    setPlayingVoice(null);
    setLoadingSample(null);
  }, []);

  const playSample = useCallback((voiceId: string) => {
    // If already playing this voice, stop it
    if (playingVoice === voiceId) {
      stopAudio();
      return;
    }

    // Stop any currently playing audio
    stopAudio();

    setLoadingSample(voiceId);

    const audio = new Audio(getVoiceSampleUrl(voiceId));
    audio.crossOrigin = 'anonymous';
    audioRef.current = audio;

    audio.addEventListener('canplaythrough', () => {
      setLoadingSample(null);
      setPlayingVoice(voiceId);
      audio.play().catch((err) => {
        console.error('Audio play failed:', err);
        setPlayingVoice(null);
      });
    }, { once: true });

    audio.addEventListener('ended', () => {
      setPlayingVoice(null);
    }, { once: true });

    audio.addEventListener('error', (e) => {
      console.error('Audio loading error:', audio.error, e);
      setLoadingSample(null);
      setPlayingVoice(null);
    }, { once: true });

    audio.load();
  }, [playingVoice, stopAudio]);

  // Cleanup audio on unmount
  useEffect(() => {
    return () => stopAudio();
  }, [stopAudio]);

  const filteredVoices = voices.filter((v) => v.gender === activeTab);

  // Group by accent
  const americanVoices = filteredVoices.filter((v) => v.accent === 'American');
  const britishVoices = filteredVoices.filter((v) => v.accent === 'British');

  if (loading) {
    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-slate-300">
          Narrator Voice
        </label>
        <div className="flex items-center gap-2 p-4 bg-slate-950 rounded-xl border border-slate-800">
          <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
          <span className="text-sm text-slate-400">Loading voices...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-slate-300">
          Narrator Voice
        </label>
        <div className="p-3 bg-red-950/30 border border-red-900/50 rounded-xl text-xs text-red-400">
          Could not load voices: {error}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
        <Mic className="w-4 h-4 text-cyan-400" />
        Narrator Voice
      </label>

      {/* Gender tabs */}
      <div className="flex gap-1 p-1 bg-slate-950 rounded-lg border border-slate-800">
        <button
          type="button"
          onClick={() => setActiveTab('male')}
          className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
            activeTab === 'male'
              ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
              : 'text-slate-400 hover:text-slate-300 border border-transparent'
          }`}
        >
          <User className="w-3 h-3" />
          Male
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('female')}
          className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
            activeTab === 'female'
              ? 'bg-purple-500/15 text-purple-300 border border-purple-500/30'
              : 'text-slate-400 hover:text-slate-300 border border-transparent'
          }`}
        >
          <Users className="w-3 h-3" />
          Female
        </button>
      </div>

      {/* Voice list */}
      <div className="max-h-[280px] overflow-y-auto pr-1 space-y-3">
        {americanVoices.length > 0 && (
          <VoiceGroup
            label="🇺🇸 American"
            voices={americanVoices}
            selectedVoice={selectedVoice}
            playingVoice={playingVoice}
            loadingSample={loadingSample}
            onSelect={onVoiceChange}
            onPlay={playSample}
          />
        )}
        {britishVoices.length > 0 && (
          <VoiceGroup
            label="🇬🇧 British"
            voices={britishVoices}
            selectedVoice={selectedVoice}
            playingVoice={playingVoice}
            loadingSample={loadingSample}
            onSelect={onVoiceChange}
            onPlay={playSample}
          />
        )}
      </div>
    </div>
  );
}

function VoiceGroup({
  label,
  voices,
  selectedVoice,
  playingVoice,
  loadingSample,
  onSelect,
  onPlay,
}: {
  label: string;
  voices: Voice[];
  selectedVoice: string;
  playingVoice: string | null;
  loadingSample: string | null;
  onSelect: (id: string) => void;
  onPlay: (id: string) => void;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1.5 px-1">
        {label}
      </div>
      <div className="grid grid-cols-1 gap-1.5">
        {voices.map((voice) => {
          const isSelected = selectedVoice === voice.id;
          const isPlaying = playingVoice === voice.id;
          const isLoading = loadingSample === voice.id;

          return (
            <div
              key={voice.id}
              className={`group flex items-center gap-2 p-2 rounded-lg border cursor-pointer transition-all ${
                isSelected
                  ? 'border-cyan-500/60 bg-cyan-950/25 shadow-[0_0_12px_rgba(6,182,212,0.08)]'
                  : 'border-slate-800 hover:border-slate-700 bg-slate-950/50'
              }`}
              onClick={() => onSelect(voice.id)}
            >
              {/* Voice avatar */}
              <div
                className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold transition-colors ${
                  isSelected
                    ? 'bg-cyan-500/20 text-cyan-300'
                    : 'bg-slate-800 text-slate-400 group-hover:text-slate-300'
                }`}
              >
                {voice.name[0]}
              </div>

              {/* Voice name */}
              <div className="flex-1 min-w-0">
                <div
                  className={`text-sm font-medium truncate transition-colors ${
                    isSelected ? 'text-cyan-200' : 'text-slate-300'
                  }`}
                >
                  {voice.name}
                </div>
              </div>

              {/* Play button */}
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onPlay(voice.id);
                }}
                className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 transition-all ${
                  isPlaying
                    ? 'bg-cyan-500 text-white'
                    : isLoading
                    ? 'bg-slate-800 text-cyan-400'
                    : 'bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700'
                }`}
                title={isPlaying ? 'Stop' : 'Preview voice'}
              >
                {isLoading ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : isPlaying ? (
                  <Square className="w-2.5 h-2.5 fill-current" />
                ) : (
                  <Play className="w-3 h-3 ml-0.5 fill-current" />
                )}
              </button>

              {/* Playing indicator — animated bars */}
              {isPlaying && (
                <div className="flex items-end gap-[2px] h-4 flex-shrink-0">
                  <span className="w-[3px] bg-cyan-400 rounded-full animate-[voice-bar_0.6s_ease-in-out_infinite]" style={{ height: '40%', animationDelay: '0ms' }} />
                  <span className="w-[3px] bg-cyan-400 rounded-full animate-[voice-bar_0.6s_ease-in-out_infinite]" style={{ height: '80%', animationDelay: '150ms' }} />
                  <span className="w-[3px] bg-cyan-400 rounded-full animate-[voice-bar_0.6s_ease-in-out_infinite]" style={{ height: '50%', animationDelay: '300ms' }} />
                  <span className="w-[3px] bg-cyan-400 rounded-full animate-[voice-bar_0.6s_ease-in-out_infinite]" style={{ height: '70%', animationDelay: '450ms' }} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
