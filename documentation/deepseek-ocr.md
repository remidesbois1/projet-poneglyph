# DeepSeek vision OCR

## Model and configuration

The integration uses the official `https://api.deepseek.com/chat/completions` endpoint and the `deepseek-flash` model (DeepSeek V4.1 Flash).
As documented on 2026-09-10, the old `deepseek-v4-flash-vision-exp` name is still accepted by DeepSeek but routes to the latest Flash model. There is no separate V4.1 vision-exp model ID to configure.

Primary references:
- https://api-docs.deepseek.com/quick_start/pricing/
- https://api-docs.deepseek.com/guides/vision/
- https://api-docs.deepseek.com/api/create-chat-completion/

Open **Clés API IA** in the site menu or **Réglages API** in the annotation sidebar. Save a personal DeepSeek key in its own card. The key can be replaced or removed independently of the Google key. No server environment variable or additional dependency is required.

Saving a key does not execute OCR. DeepSeek uses the account associated with that key, and calls are billed by DeepSeek.

## Annotation tools

DeepSeek appears twice in the *OCR de la page* list of the annotation sidebar; select either or both.

- **Bulles:** reads each bubble crop with the shared `ocr_bubble` prompt, for page OCR and for hand-drawn boxes. It also appears under *Relire via API* in the bubble editor.
- **Page entière:** one call reads the whole page with the shared `ocr_page_bbox` / `strict_json_suffix` prompts. The whole response must validate first: `content` is text and `bbox` is `[x1,y1,x2,y2]`, normalized to 0–1000, with positive dimensions.
- *Lancer l’OCR de la page* runs the YOLO detector, which is the truth: DeepSeek page boxes are rallied one-to-one onto YOLO boxes (IoU > 0.05, as in the batch OCR), unmatched DeepSeek boxes are dropped and the review follows the YOLO reading order. Each bubble then opens in the editor with every candidate; nothing is saved without validation. Hand-drawn boxes only run the bubble models.
- Without a key, the sidebar button becomes *Configurer l’accès API* and no request is sent. Both modes work in the sandbox, where annotations remain local.
- Navigating away or cancelling aborts pending DeepSeek requests. Page identity and image identity are checked again before results are used. Background bubble results are reused during review instead of calling a paid model twice.

Gemini descriptions, embeddings, search and import tools are unchanged.

## Data flow and failure handling

The key is stored under `deepseek_api_key` in browser local storage, separately from `google_api_key`. Like the existing Google key, it is accessible to scripts executing on this origin; it is not an encrypted key vault.

Requests go through the same-origin `/api/deepseek` route so the browser does not depend on upstream CORS. The route forwards the user-supplied key transiently as an Authorization header to the fixed official endpoint. It does not use a server-owned key, persist the key or image, forward cookies or Supabase tokens, log private payloads, or follow redirects. Do not enable proxy logging of request headers/bodies in deployment.

Application limits are deliberately smaller than the provider's maximum: one image, JPEG/PNG/WebP, 10 MiB decoded, 8192 pixels per side, 24 megapixels, 1 MiB upstream response. Thinking is disabled for OCR and bbox calls request JSON output. The timeout is 120 seconds on the server, with a slightly longer browser timeout. There are no automatic paid retries or fallback calls to Gemini.

Invalid keys, insufficient balance, rate limits, provider unavailability, truncation, invalid JSON and invalid coordinates produce explicit errors. Partial database imports are reported separately and successful creations are retained. Native API image processing can still resize inputs internally; `detail: original` does not guarantee native-resolution inference.

## Validation

Unit/integration tests cover the BYOK proxy, request/response limits, provider errors, parsing, key isolation, model selection, cancellation, duplicate-request prevention and both annotation lifecycles. Browser QA must intercept providers and database writes and use fixture images. No live paid request should be made with a developer's existing key during automated tests.
