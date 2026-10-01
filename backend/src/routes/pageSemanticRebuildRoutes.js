const { rebuildPageSemanticData, SemanticRebuildError } = require('../services/pageSemanticRebuild');

function ensureSemanticRebuildAvailable(userGeminiApiKey, environment = process.env) {
  if (!environment.VOYAGE_API_KEY || (!environment.GOOGLE_API_KEY && !userGeminiApiKey)) {
    throw new SemanticRebuildError('SEMANTIC_REBUILD_UNAVAILABLE', 'Configurez la clé serveur Voyage et une clé Gemini serveur ou personnelle dans votre profil avant de reconstruire les embeddings.', 503);
  }
}

function readUserGeminiApiKey(req) {
  const rawKey = req.get?.('X-Gemini-API-Key') ?? req.headers?.['x-gemini-api-key'];
  if (rawKey === undefined || rawKey === '') return undefined;
  if (typeof rawKey !== 'string' || !/^[A-Za-z0-9_-]{10,256}$/.test(rawKey.trim())) {
    throw new SemanticRebuildError('INVALID_GEMINI_API_KEY', 'Clé Gemini personnelle invalide. Vérifiez-la dans votre profil.', 400);
  }
  return rawKey.trim();
}

function createPageSemanticRebuildHandler(dependencies) {
  return async (req, res) => {
    const rawId = req.body?.id_page;
    const idPage = typeof rawId === 'number' || (typeof rawId === 'string' && /^\d+$/.test(rawId))
      ? Number(rawId) : NaN;
    const mangaSlug = req.query?.manga;
    if (!Number.isSafeInteger(idPage) || idPage <= 0
      || typeof mangaSlug !== 'string' || !/^[a-z0-9][a-z0-9_-]{0,119}$/i.test(mangaSlug)
      || !req.body?.description || typeof req.body.description !== 'object' || Array.isArray(req.body.description)) {
      return res.status(400).json({ code: 'INVALID_REQUEST', error: 'Page, manga et description JSON requis.' });
    }
    try {
      const userGeminiApiKey = readUserGeminiApiKey(req);
      dependencies.ensureAvailable?.(userGeminiApiKey);
      const result = await rebuildPageSemanticData({
        ...dependencies, idPage, mangaSlug, description: req.body.description, userGeminiApiKey,
      });
      return res.json(result);
    } catch (error) {
      if (error instanceof SemanticRebuildError) {
        return res.status(error.statusCode).json({ code: error.code, error: error.message });
      }
      if (['42703', 'PGRST202', 'PGRST204'].includes(error.code)) {
        return res.status(503).json({ code: 'SEMANTIC_REBUILD_UNAVAILABLE', error: 'Appliquez la migration SQL du rebuild Astra avant de reprendre.' });
      }
      if (error.code === 'GEMINI_USER_KEY_REJECTED') {
        return res.status(503).json({ code: 'GEMINI_USER_KEY_REJECTED', error: 'La clé Gemini personnelle a été refusée. Vérifiez-la dans votre profil puis reprenez ; la description Astra est conservée.' });
      }
      if (error.response?.status === 429) {
        const message = error.geminiKeySource === 'user'
          ? 'Quota de la clé Gemini personnelle atteint. Reprenez après son rétablissement ; la description Astra est conservée localement.'
          : error.geminiKeySource === 'server'
            ? 'Quota de la clé Gemini serveur atteint. Ajoutez une autre clé Gemini dans votre profil puis reprenez ; la description Astra est conservée localement.'
            : 'Quota du service d’embeddings atteint. Reprenez plus tard ; la description Astra est conservée localement.';
        return res.status(429).json({ code: 'EMBEDDING_QUOTA_EXCEEDED', error: message });
      }
      if ([401, 403].includes(error.response?.status)) {
        return res.status(503).json({ code: 'SEMANTIC_REBUILD_UNAVAILABLE', error: 'Accès au service d’embeddings refusé. Vérifiez la configuration serveur.' });
      }
      return res.status(502).json({ code: 'SEMANTIC_REBUILD_FAILED', error: 'La reconstruction des embeddings a échoué. Les anciennes données sont conservées et la description Astra pourra être réutilisée.' });
    }
  };
}

module.exports = { createPageSemanticRebuildHandler, readUserGeminiApiKey, ensureSemanticRebuildAvailable };
