/*
  # Create manga-pdfs storage bucket

  1. Storage
    - Creates a storage bucket named 'manga-pdfs' for PDF uploads
  2. Security
    - Enables public access for reading (for demo purposes)
    - Authenticated users can upload
*/

-- Create the storage bucket
INSERT INTO storage.buckets (id, name, public)
VALUES ('manga-pdfs', 'manga-pdfs', true)
ON CONFLICT (id) DO NOTHING;

-- Allow public read access
CREATE POLICY "Public read access"
  ON storage.objects FOR SELECT
  TO public
  USING (bucket_id = 'manga-pdfs');

-- Allow authenticated users to upload
CREATE POLICY "Authenticated users can upload"
  ON storage.objects FOR INSERT
  TO authenticated
  WITH CHECK (bucket_id = 'manga-pdfs');