-- 2026-11-04: Drop Modal OCR quotas and training provider columns
-- Safe to execute against existing databases migrating away from Modal.

-- 1. Drop Modal OCR quota tracking objects
DROP FUNCTION IF EXISTS public.consume_modal_ocr_quota(uuid, text, text, text, integer, integer, integer, integer, integer, integer, integer, integer, integer, integer);
DROP FUNCTION IF EXISTS public.consume_modal_ocr_quota(uuid, text, text, integer, integer, integer, integer, integer, integer);
DROP TABLE IF EXISTS public.modal_ocr_quota_buckets;

-- 2. Drop Modal-specific columns from training_jobs if they exist
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'training_jobs' AND column_name = 'modal_function_name'
    ) THEN
        ALTER TABLE public.training_jobs DROP COLUMN IF EXISTS modal_function_name;
        ALTER TABLE public.training_jobs DROP COLUMN IF EXISTS modal_call_id;
        ALTER TABLE public.training_jobs DROP COLUMN IF EXISTS modal_volume_name;
    END IF;
END $$;

-- 3. Update provider default and check constraint on training_jobs
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'training_jobs' AND column_name = 'provider'
    ) THEN
        ALTER TABLE public.training_jobs ALTER COLUMN provider SET DEFAULT 'local';
        ALTER TABLE public.training_jobs DROP CONSTRAINT IF EXISTS training_jobs_provider_check;
        -- Validate future writes without failing historical rows
        ALTER TABLE public.training_jobs ADD CONSTRAINT training_jobs_provider_check
            CHECK (provider IN ('runpod', 'local')) NOT VALID;
    END IF;
END $$;
