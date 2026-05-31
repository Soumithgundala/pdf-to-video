/*
  # Manga Recap Video Pipeline Schema

  1. New Tables
    - `jobs` - Main job tracking table
      - `id` (uuid, primary key)
      - `user_id` (uuid, references auth.users)
      - `status` (text: pending, processing, completed, failed)
      - `pdf_filename` (text)
      - `pdf_path` (text)
      - `total_pages` (integer)
      - `total_panels` (integer)
      - `error_message` (text, nullable)
      - `created_at` (timestamp)
      - `updated_at` (timestamp)
      - `completed_at` (timestamp, nullable)
    
    - `pages` - Page images extracted from PDF
      - `id` (uuid, primary key)
      - `job_id` (uuid, references jobs)
      - `page_number` (integer)
      - `image_path` (text)
      - `created_at` (timestamp)
    
    - `panels` - Individual panels extracted from pages
      - `id` (uuid, primary key)
      - `job_id` (uuid, references jobs)
      - `page_id` (uuid, references pages)
      - `panel_id` (integer - global sequential ID like P1, P2, etc)
      - `image_path` (text)
      - `bbox_x` (integer)
      - `bbox_y` (integer)
      - `bbox_width` (integer)
      - `bbox_height` (integer)
      - `created_at` (timestamp)
    
    - `video_parts` - Generated video parts (4 per job)
      - `id` (uuid, primary key)
      - `job_id` (uuid, references jobs)
      - `part_number` (integer, 1-4)
      - `script` (text)
      - `selected_panels` (jsonb - array of panel IDs)
      - `audio_path` (text, nullable)
      - `audio_duration_ms` (integer, nullable)
      - `video_path` (text, nullable)
      - `status` (text: pending, processing, completed, failed)
      - `created_at` (timestamp)
      - `updated_at` (timestamp)
    
    - `contact_sheets` - Generated contact sheets
      - `id` (uuid, primary key)
      - `job_id` (uuid, references jobs)
      - `sheet_number` (integer)
      - `image_path` (text)
      - `panel_start_id` (integer)
      - `panel_end_id` (integer)
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on all tables
    - Users can only access their own jobs and related data
    - Service role has full access for backend processing
*/

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  pdf_filename text NOT NULL,
  pdf_path text NOT NULL,
  total_pages integer DEFAULT 0,
  total_panels integer DEFAULT 0,
  error_message text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  completed_at timestamptz
);

-- Pages table
CREATE TABLE IF NOT EXISTS pages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id uuid REFERENCES jobs(id) ON DELETE CASCADE NOT NULL,
  page_number integer NOT NULL,
  image_path text NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- Panels table
CREATE TABLE IF NOT EXISTS panels (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id uuid REFERENCES jobs(id) ON DELETE CASCADE NOT NULL,
  page_id uuid REFERENCES pages(id) ON DELETE CASCADE NOT NULL,
  panel_id integer NOT NULL,
  image_path text NOT NULL,
  bbox_x integer NOT NULL,
  bbox_y integer NOT NULL,
  bbox_width integer NOT NULL,
  bbox_height integer NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- Video parts table
CREATE TABLE IF NOT EXISTS video_parts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id uuid REFERENCES jobs(id) ON DELETE CASCADE NOT NULL,
  part_number integer NOT NULL CHECK (part_number BETWEEN 1 AND 4),
  script text NOT NULL,
  selected_panels jsonb NOT NULL DEFAULT '[]',
  audio_path text,
  audio_duration_ms integer,
  video_path text,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Contact sheets table
CREATE TABLE IF NOT EXISTS contact_sheets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id uuid REFERENCES jobs(id) ON DELETE CASCADE NOT NULL,
  sheet_number integer NOT NULL,
  image_path text NOT NULL,
  panel_start_id integer NOT NULL,
  panel_end_id integer NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- Enable RLS
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE panels ENABLE ROW LEVEL SECURITY;
ALTER TABLE video_parts ENABLE ROW LEVEL SECURITY;
ALTER TABLE contact_sheets ENABLE ROW LEVEL SECURITY;

-- Jobs policies
CREATE POLICY "Users can view own jobs"
  ON jobs FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can create own jobs"
  ON jobs FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own jobs"
  ON jobs FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Pages policies
CREATE POLICY "Users can view pages for own jobs"
  ON pages FOR SELECT
  TO authenticated
  USING (EXISTS (SELECT 1 FROM jobs WHERE jobs.id = pages.job_id AND jobs.user_id = auth.uid()));

CREATE POLICY "Users can create pages for own jobs"
  ON pages FOR INSERT
  TO authenticated
  WITH CHECK (EXISTS (SELECT 1 FROM jobs WHERE jobs.id = pages.job_id AND jobs.user_id = auth.uid()));

-- Panels policies
CREATE POLICY "Users can view panels for own jobs"
  ON panels FOR SELECT
  TO authenticated
  USING (EXISTS (SELECT 1 FROM jobs WHERE jobs.id = panels.job_id AND jobs.user_id = auth.uid()));

CREATE POLICY "Users can create panels for own jobs"
  ON panels FOR INSERT
  TO authenticated
  WITH CHECK (EXISTS (SELECT 1 FROM jobs WHERE jobs.id = panels.job_id AND jobs.user_id = auth.uid()));

-- Video parts policies
CREATE POLICY "Users can view video parts for own jobs"
  ON video_parts FOR SELECT
  TO authenticated
  USING (EXISTS (SELECT 1 FROM jobs WHERE jobs.id = video_parts.job_id AND jobs.user_id = auth.uid()));

CREATE POLICY "Users can create video parts for own jobs"
  ON video_parts FOR INSERT
  TO authenticated
  WITH CHECK (EXISTS (SELECT 1 FROM jobs WHERE jobs.id = video_parts.job_id AND jobs.user_id = auth.uid()));

CREATE POLICY "Users can update video parts for own jobs"
  ON video_parts FOR UPDATE
  TO authenticated
  USING (EXISTS (SELECT 1 FROM jobs WHERE jobs.id = video_parts.job_id AND jobs.user_id = auth.uid()))
  WITH CHECK (EXISTS (SELECT 1 FROM jobs WHERE jobs.id = video_parts.job_id AND jobs.user_id = auth.uid()));

-- Contact sheets policies
CREATE POLICY "Users can view contact sheets for own jobs"
  ON contact_sheets FOR SELECT
  TO authenticated
  USING (EXISTS (SELECT 1 FROM jobs WHERE jobs.id = contact_sheets.job_id AND jobs.user_id = auth.uid()));

CREATE POLICY "Users can create contact sheets for own jobs"
  ON contact_sheets FOR INSERT
  TO authenticated
  WITH CHECK (EXISTS (SELECT 1 FROM jobs WHERE jobs.id = contact_sheets.job_id AND jobs.user_id = auth.uid()));

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_pages_job_id ON pages(job_id);
CREATE INDEX IF NOT EXISTS idx_panels_job_id ON panels(job_id);
CREATE INDEX IF NOT EXISTS idx_panels_page_id ON panels(page_id);
CREATE INDEX IF NOT EXISTS idx_video_parts_job_id ON video_parts(job_id);
CREATE INDEX IF NOT EXISTS idx_contact_sheets_job_id ON contact_sheets(job_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_jobs_updated_at
  BEFORE UPDATE ON jobs
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_video_parts_updated_at
  BEFORE UPDATE ON video_parts
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
