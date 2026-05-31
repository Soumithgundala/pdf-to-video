import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

const isPlaceholder = !supabaseUrl || supabaseUrl.includes('zmkmotymomcbhjomdkmb') || supabaseUrl.includes('placeholder');

export const supabase = isPlaceholder 
  ? {
      auth: {
        getSession: async () => {
          const mockUser = {
            id: 'local-guest',
            email: 'guest@local.manga',
            app_metadata: {},
            user_metadata: {},
            aud: 'authenticated',
            created_at: new Date().toISOString()
          };
          return {
            data: {
              session: {
                access_token: 'mock-token',
                token_type: 'bearer',
                expires_in: 3600,
                refresh_token: 'mock-refresh',
                user: mockUser,
              }
            },
            error: null
          };
        },
        onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
        signUp: async (credentials: any) => {
          const mockUser = { id: 'local-guest', email: credentials.email };
          return { data: { user: mockUser, session: null }, error: null };
        },
        signInWithPassword: async (credentials: any) => {
          const mockUser = { id: 'local-guest', email: credentials.email };
          return { data: { user: mockUser, session: null }, error: null };
        },
        signOut: async () => ({ error: null }),
      }
    } as any
  : createClient(supabaseUrl, supabaseAnonKey);

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface Job {
  id: string;
  user_id: string;
  status: JobStatus;
  pdf_filename: string;
  pdf_path: string;
  total_pages: number;
  total_panels: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface VideoPart {
  id: string;
  job_id: string;
  part_number: number;
  script: string;
  selected_panels: string[];
  audio_path: string | null;
  audio_duration_ms: number | null;
  video_path: string | null;
  status: JobStatus;
  created_at: string;
  updated_at: string;
}

export interface Panel {
  id: string;
  job_id: string;
  page_id: string;
  panel_id: number;
  image_path: string;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  created_at: string;
}

export interface ContactSheet {
  id: string;
  job_id: string;
  sheet_number: number;
  image_path: string;
  panel_start_id: number;
  panel_end_id: number;
  created_at: string;
}
