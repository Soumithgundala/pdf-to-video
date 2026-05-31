import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2.30.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Client-Info, Apikey",
};

const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  const url = new URL(req.url);
  const path = url.pathname.replace("/functions/v1/manga-api", "");

  try {
    // Health check
    if (path === "/health" && req.method === "GET") {
      return new Response(
        JSON.stringify({ status: "ok", timestamp: new Date().toISOString() }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Upload PDF - creates a job record
    if (path === "/api/jobs/upload" && req.method === "POST") {
      const formData = await req.formData();
      const file = formData.get("file") as File | null;

      if (!file) {
        return new Response(
          JSON.stringify({ detail: "No file provided" }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // Generate job ID
      const jobId = crypto.randomUUID();
      const userId = "anonymous";

      // Create job record
      const { error } = await supabase.from("jobs").insert({
        id: jobId,
        user_id: userId,
        status: "pending",
        pdf_filename: file.name,
        pdf_path: `uploads/${jobId}/${file.name}`,
      });

      if (error) {
        return new Response(
          JSON.stringify({ detail: error.message }),
          { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // Upload PDF to storage
      const fileBuffer = await file.arrayBuffer();
      const { error: storageError } = await supabase.storage
        .from("manga-pdfs")
        .upload(`${jobId}/${file.name}`, fileBuffer, {
          contentType: "application/pdf",
        });

      if (storageError) {
        console.error("Storage error:", storageError);
        // Continue anyway - job record is created
      }

      return new Response(
        JSON.stringify({ job_id: jobId }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Process job - updates status
    if (path.match(/^\/api\/jobs\/[^/]+\/process$/) && req.method === "POST") {
      const jobId = path.split("/")[3];

      // Get job
      const { data: job, error: jobError } = await supabase
        .from("jobs")
        .select("*")
        .eq("id", jobId)
        .single();

      if (jobError || !job) {
        return new Response(
          JSON.stringify({ detail: "Job not found" }),
          { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      if (job.status !== "pending") {
        return new Response(
          JSON.stringify({ detail: "Job already processed or processing" }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // Update status to processing
      const { error: updateError } = await supabase
        .from("jobs")
        .update({
          status: "processing",
          updated_at: new Date().toISOString(),
        })
        .eq("id", jobId);

      if (updateError) {
        return new Response(
          JSON.stringify({ detail: updateError.message }),
          { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // Since we can't run the Python pipeline in edge functions,
      // we'll simulate the processing for demo purposes
      setTimeout(async () => {
        // Simulate processing time
        await new Promise((resolve) => setTimeout(resolve, 3000));

        // Create mock video parts
        const parts = [1, 2, 3, 4];
        for (const partNumber of parts) {
          await supabase.from("video_parts").insert({
            job_id: jobId,
            part_number: partNumber,
            script: `Demo script for part ${partNumber}. This is a placeholder for the actual voiceover script that would be generated from analyzing the manga content.`,
            selected_panels: ["P1", "P2", "P3", "P4", "P5"],
            status: "completed",
          });
        }

        // Update job as completed
        await supabase
          .from("jobs")
          .update({
            status: "completed",
            total_pages: 10,
            total_panels: 40,
            completed_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          })
          .eq("id", jobId);
      }, 100);

      return new Response(
        JSON.stringify({ status: "processing", job_id: jobId }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Get job status
    if (path.match(/^\/api\/jobs\/[^/]+\/status$/) && req.method === "GET") {
      const jobId = path.split("/")[3];

      const { data: job, error } = await supabase
        .from("jobs")
        .select("*")
        .eq("id", jobId)
        .single();

      if (error || !job) {
        return new Response(
          JSON.stringify({ detail: "Job not found" }),
          { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      let progress = 0;
      if (job.status === "processing") {
        progress = 0.5;
      } else if (job.status === "completed") {
        progress = 1.0;
      }

      return new Response(
        JSON.stringify({ job_id: jobId, status: job.status, progress }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Get job details
    if (path.match(/^\/api\/jobs\/[^/]+$/) && req.method === "GET") {
      const jobId = path.split("/")[3];

      const { data: job, error: jobError } = await supabase
        .from("jobs")
        .select("*")
        .eq("id", jobId)
        .single();

      if (jobError || !job) {
        return new Response(
          JSON.stringify({ detail: "Job not found" }),
          { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      const { data: videoParts } = await supabase
        .from("video_parts")
        .select("*")
        .eq("job_id", jobId);

      return new Response(
        JSON.stringify({ job, video_parts: videoParts || [] }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    return new Response(
      JSON.stringify({ detail: "Not found" }),
      { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error:", error);
    return new Response(
      JSON.stringify({ detail: String(error) }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
