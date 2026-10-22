---
title: Resume Screener API
sdk: docker
app_port: 8000
---

## Deploy the backend

1. Create a private Hugging Face Docker Space.
2. Copy the **contents** of this repository's `backend/` directory into the
   Space repository root, including its Dockerfile.
3. Create a read-only Hugging Face token with access to your private model repo
   and add it as the `HF_TOKEN` Space secret.
4. Set `FINETUNED_REPO`, optionally pin `FINETUNED_REVISION` to an approved
   commit, and configure `TRUSTED_PROXY_COUNT` and `ALLOWED_ORIGINS`.
5. Restart the Space. Training runs locally only; upload the trained model and
   `calibrator.pkl` to the private repo before restarting to deploy an update.
