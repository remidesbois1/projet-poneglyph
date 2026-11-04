# Training job tracking

Fine-tuning pipelines can record their job state in Supabase.

## Database schema

Apply this migration to Supabase before tracking training jobs:

```text
backend/sql/2026-07-01_add_training_jobs.sql
```

You can verify the migration from the backend package:

```powershell
cd backend
npm run training:schema -- --check
```

To apply it through the Supabase Management API, provide a personal access token with database write access:

```powershell
cd backend
$env:SUPABASE_ACCESS_TOKEN="..."
npm run training:schema
```

The training helpers in `common_training` use `training_jobs` and `model_versions` to persist job state and generated model-version metadata. Each pipeline records the provider it runs on (`local` or `runpod`), read from the `TRAINING_PROVIDER` environment variable.
