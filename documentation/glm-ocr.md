# GLM 5.3 vision OCR (hébergement Mistral)

## Model and configuration

The integration uses the official Mistral endpoint `https://api.mistral.ai/v1/chat/completions` with the `zai-glm-5-3-flash` model (Z.ai GLM 5.3 Flash, hosted by Mistral). As documented on 2026-11-04, only the Flash variant of GLM 5.3 accepts images on Mistral; the text-only `zai-glm-5-3` cannot serve OCR. The model is served by Mistral without modification.

Primary references:
- https://docs.mistral.ai/models/zai-glm-5-3-flash
- https://docs.mistral.ai/api/endpoint/chat

Open **Clés API IA** in the site menu or **Réglages API** in the annotation sidebar. Save a personal Mistral key in its own card. The key can be replaced or removed independently of the Google, DeepSeek and ChatGPT credentials. No server environment variable or additional dependency is required.

Saving a key does not execute OCR. Mistral uses the account associated with that key, and calls are billed by Mistral.

## Annotation tools

GLM 5.3 Flash appears once in the *OCR de la page* list of the annotation sidebar, under **Page entière**. It never reads individual bubbles.

- **Page entière:** one call reads the whole page with the shared `ocr_page_bbox` / `strict_json_suffix` prompts. The whole response must validate first: `content` is text and `bbox` is `[x1,y1,x2,y2]`, normalized to 0–1000, with positive dimensions.
- GLM 5.3 Flash does not reliably honor a 0–1000 normalization: about half of the measured calls returned bubble coordinates on a larger, unstable scale (up to ~1500 on a 1500×2349 page, with no linear correction factor). The route therefore asks for **pixel coordinates with the image's explicit dimensions** (measured 12/12 in-bounds) and the client rescales each axis back to the pipeline's 0–1000 convention using the source image's pixel size; residual out-of-bounds values are clamped, a bubble whose geometry stays invalid after rescaling is dropped on its own, and misplaced boxes are ignored by the YOLO rally anyway. The model also sometimes answers with content chunks (reasoning + text) instead of a plain string; the server route extracts the text chunk, and answers without any text chunk are rejected.
- *Lancer l’OCR de la page* runs the YOLO detector, which is the truth: GLM page boxes are rallied one-to-one onto YOLO boxes (IoU > 0.05, as in the batch OCR), unmatched GLM boxes are dropped and the review follows the YOLO reading order. Each bubble then opens in the editor with every candidate; nothing is saved without validation.

## Data flow and failure handling

The key is stored under `mistral_api_key` in browser local storage, separately from `google_api_key` and `deepseek_api_key`. Like the existing keys, it is accessible to scripts executing on this origin; it is not an encrypted key vault.

Requests go through the same-origin `/api/mistral` route so the browser does not depend on upstream CORS. The route forwards the user-supplied key transiently as an Authorization header to the fixed official endpoint. It does not use a server-owned key, persist the key or image, forward cookies or Supabase tokens, log private payloads, or follow redirects. Do not enable proxy logging of request headers/bodies in deployment.

Application limits are deliberately smaller than the provider's maximum: one image, JPEG/PNG/WebP, 10 MiB decoded, 8192 pixels per side, 24 megapixels, 1 MiB upstream response. Every call requests JSON output with `reasoning_effort: low`. The timeout is 120 seconds on the server, with a slightly longer browser timeout. There are no automatic paid retries or fallback calls to another provider.

Invalid keys, insufficient credits, rate limits, provider unavailability, truncation, invalid JSON and invalid coordinates produce explicit errors. Partial database imports are reported separately and successful creations are retained.

## Validation

Unit/integration tests cover the BYOK proxy, request/response limits, provider errors, parsing, key isolation and model selection. Browser QA must intercept providers and database writes and use fixture images. No live paid request should be made with a developer's existing key during automated tests.
