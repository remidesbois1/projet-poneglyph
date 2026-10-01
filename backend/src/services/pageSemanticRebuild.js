const { ASTRA_DESCRIPTION_MODEL, DESCRIPTION_PROMPT_VERSION, parsePageDescription } = require('@poneglyph/shared/page-description');
const { buildPageEmbeddingText } = require('../utils/pageEmbeddingText');

class SemanticRebuildError extends Error {
  constructor(code, message, statusCode = 500) {
    super(message);
    this.code = code;
    this.statusCode = statusCode;
  }
}

function validateEmbedding(embedding, dimension, provider) {
  if (!Array.isArray(embedding) || embedding.length !== dimension
    || !embedding.every((value) => typeof value === 'number' && Number.isFinite(value))
    || !embedding.some((value) => value !== 0)) {
    throw new SemanticRebuildError('INVALID_EMBEDDING', `Embedding ${provider} invalide (${dimension} valeurs attendues).`, 502);
  }
  return embedding;
}

async function rebuildPageSemanticData({ db, idPage, mangaSlug, description, generateVoyageEmbedding, generateGeminiEmbedding, userGeminiApiKey, now = () => new Date() }) {
  let validated;
  try {
    validated = parsePageDescription(description);
  } catch (error) {
    throw new SemanticRebuildError('INVALID_DESCRIPTION', error.message, 400);
  }
  const { data: page, error: readError } = await db.from('pages')
    .select('id, url_image, bulles(texte_propose, statut, order), chapitres!inner(tomes!inner(mangas!inner(slug)))')
    .eq('id', idPage)
    .eq('chapitres.tomes.mangas.slug', mangaSlug)
    .maybeSingle();
  if (readError) throw readError;
  if (!page) throw new SemanticRebuildError('PAGE_NOT_FOUND', 'Page introuvable dans ce manga.', 404);
  if (!page.url_image) throw new SemanticRebuildError('ORIGINAL_IMAGE_MISSING', 'Image originale manquante.', 422);

  const bulles = [...(page.bulles || [])].sort((left, right) => (left.order ?? Infinity) - (right.order ?? Infinity));
  const text = buildPageEmbeddingText({ description: validated, bulles });
  const voyage = validateEmbedding(await generateVoyageEmbedding(text, 'document'), 1024, 'Voyage');
  // url_image is a private storage reference; the Gemini helper reads it through readPageImage.
  let usedUserGeminiKey = false;
  const geminiOptions = { requireImage: true };
  if (userGeminiApiKey) {
    geminiOptions.fallbackApiKey = userGeminiApiKey;
    geminiOptions.onFallback = () => { usedUserGeminiKey = true; };
  }
  const gemini = validateEmbedding(await generateGeminiEmbedding(text, 'RETRIEVAL_DOCUMENT', page.url_image, geminiOptions), 3072, 'Gemini');

  const provenance = {
    description_model: ASTRA_DESCRIPTION_MODEL,
    description_prompt_version: DESCRIPTION_PROMPT_VERSION,
    description_generated_at: now().toISOString(),
  };
  // One PostgreSQL UPDATE publishes the description, all vectors and the completion marker together.
  const { data: saved, error: saveError } = await db.from('pages').update({
    description: validated,
    embedding_voyage: voyage,
    embedding_gemini: gemini,
    // An old F2LLM vector describes the previous description and must not stay searchable.
    embedding_f2llm: null,
    ...provenance,
  }).eq('id', idPage).eq('url_image', page.url_image).select('id').maybeSingle();
  if (saveError) throw saveError;
  if (!saved) throw new SemanticRebuildError('PAGE_CHANGED', 'La page a changé pendant le traitement. Reprenez cette page.', 409);
  return { success: true, id_page: saved.id, ...provenance, ...(usedUserGeminiKey ? { gemini_key_source: 'user' } : {}) };
}

module.exports = { SemanticRebuildError, rebuildPageSemanticData };
