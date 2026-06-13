const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;
const API_URL = import.meta.env.VITE_API_URL || `${SUPABASE_URL}/functions/v1/manga-api`;

export async function uploadPDF(file: File): Promise<{ job_id: string }> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_URL}/api/jobs/upload`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(error.detail || 'Upload failed');
  }

  return response.json();
}

export async function processJob(
  jobId: string,
  options?: { colorizerMode?: string; llmProvider?: string }
): Promise<void> {
  const response = await fetch(`${API_URL}/api/jobs/${jobId}/process`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      llm_provider: options?.llmProvider || 'google',
      colorizer_mode: options?.colorizerMode || 'stable_diffusion',
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Processing failed' }));
    throw new Error(error.detail || 'Processing failed');
  }
}

export async function getJobStatus(jobId: string): Promise<any> {
  const response = await fetch(`${API_URL}/api/jobs/${jobId}/status`, {
    headers: {
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
    },
  });

  if (!response.ok) {
    throw new Error('Failed to get job status');
  }

  return response.json();
}

export function getVideoUrl(jobId: string, partNumber: number): string {
  return `${API_URL}/api/jobs/${jobId}/videos/${partNumber}`;
}

export async function uploadCharacterReferences(jobId: string, files: File[]): Promise<void> {
  if (files.length === 0) return;
  
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });

  const response = await fetch(`${API_URL}/api/jobs/${jobId}/character-references`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Uploading character references failed' }));
    throw new Error(error.detail || 'Uploading character references failed');
  }
}
