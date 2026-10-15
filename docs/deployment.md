# Deployment

## Vercel frontend

The root [`vercel.json`](../vercel.json) keeps the repository root as the
Vercel project directory and configures these commands:

```text
installCommand: npm --prefix frontend ci
buildCommand: npm --prefix frontend run build
outputDirectory: frontend/dist
```

It also rewrites client-side routes to `/index.html`. Configure
`VITE_API_URL` in the Vercel Preview and Production environments as appropriate;
Vite embeds this value at build time, so redeploy after changing it.

If the preview build does not work with the root project configuration, use
this dashboard fallback:

1. Set the Vercel project's **Root Directory** to `frontend`.
2. Keep **Include source files outside of the Root Directory** disabled.
3. Use `npm ci` for Install Command and `npm run build` for Build Command, with
   `dist` as the output directory.
4. Keep `frontend/vercel.json` in place for the SPA rewrite.

## Backend hosting

Backend hosting is not configured yet, and the repository has no backend
Dockerfile or deployment configuration. When the API is hosted, configure its
`ALLOWED_ORIGINS` environment variable with the exact browser origins that may
call it. The frontend's `VITE_API_URL` must point to that public API base URL.
See [local testing with ngrok](local-testing-with-ngrok.md) for a temporary
preview-testing setup.
