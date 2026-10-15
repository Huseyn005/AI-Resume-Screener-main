# Local API testing from a Vercel preview

This exposes a local FastAPI instance temporarily so a deployed frontend
preview can send a real screening request.

1. Set `ALLOWED_ORIGINS` in `backend/.env` to include the exact Vercel preview
   origin (scheme and host, without a trailing path). Keep any localhost
   origins needed for local frontend development.
2. Start the API from `backend/`:

   ```sh
   uvicorn app.main:app --port 8000
   ```

3. In another terminal, start the tunnel:

   ```sh
   ngrok http 8000
   ```

4. Copy the HTTPS forwarding URL printed by ngrok, open it in a browser once,
   and pass the ngrok warning page before using the API from the app.
5. Set the Vercel Preview environment variable `VITE_API_URL` to that HTTPS
   forwarding URL, then redeploy the preview.
6. Open the preview and submit a screening request. The API CORS configuration
   must allow the preview's origin.

The ngrok URL changes after each ngrok restart. Update `VITE_API_URL` and
redeploy whenever it changes. This is a temporary testing approach, not a
backend hosting setup.
