const express = require('express');
const router = express.Router();
const { authMiddleware, roleCheck } = require('../middleware/auth');

const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const mime = require('mime-types');
const inputLimits = require('@poneglyph/shared/input-limits.json');

const { supabaseAdmin } = require('../config/supabaseClient');
const { logBubbleHistory } = require('../utils/auditLogger');
const { clearBubbleGeometryCache } = require('../utils/bubbleGeometry');
const { generateGeminiEmbedding } = require('../utils/geminiClient');
const { generateVoyageEmbedding } = require('../utils/voyageClient');
const { generateF2llmEmbedding } = require('../utils/f2llmClient');
const {
  PROMPT_KEYS,
  PROMPT_CONTENT_MAX_LENGTH,
  getDefaultPrompt,
  getPromptRegistry,
  getPromptContents,
  invalidatePromptCache,
} = require('../utils/promptRegistry');
const { buildPageEmbeddingText } = require('../utils/pageEmbeddingText');
const {
  PageStorageError,
  createPageStorageRef,
  getPrivatePagesBucketName,
  normalizePageStorageKey,
} = require('../utils/pageStorage');
const { getPageImagePath } = require('../utils/publicMedia');
const { isPageImageValidationError, preparePageUpload } = require('../services/pageUpload');
const { chapterUploadBodySchema, validateRequest } = require('../validation/requestSchemas');
const {
  chapterArchiveUploadMiddleware,
  createChapterImportHandlers,
} = require('./chapterImportRoutes');

// Ensure environment variables are loaded
if (process.env.NODE_ENV !== 'production') {
  require('dotenv').config();
}

const r2Config = {
  region: 'auto',
  endpoint: process.env.R2_ENDPOINT,
  forcePathStyle: true,
  credentials: (process.env.R2_ACCESS_KEY_ID && process.env.R2_SECRET_ACCESS_KEY) ? {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  } : undefined,
};

// Log warning if variables are missing
if (!process.env.R2_ENDPOINT || !process.env.R2_ACCESS_KEY_ID || !process.env.R2_SECRET_ACCESS_KEY) {
  console.warn('[AdminRoutes] Warning: R2 environment variables are missing. S3 client might not work correctly.');
}

const s3Client = new S3Client(r2Config);
const BUCKET_NAME = process.env.R2_BUCKET_NAME;
const PUBLIC_URL_BASE = process.env.R2_PUBLIC_URL;

const UPLOAD_DIR = 'temp_uploads/';
if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR);
}

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, UPLOAD_DIR);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });
const pageUpload = multer({
  storage: storage,
  limits: {
    fileSize: inputLimits.pageImageBytes,
    files: 1,
    fields: 1,
    parts: 2,
    fieldNameSize: 100,
    fieldSize: 2048,
  },
});

function uploadSinglePage(req, res, next) {
  pageUpload.single('file')(req, res, (error) => {
    if (!error) return next();
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({ error: 'L’image de page dépasse la taille maximale autorisée.' });
    }
    if (String(error.code || '').startsWith('LIMIT_')) {
      return res.status(400).json({ error: 'Le formulaire d’upload de page est invalide.' });
    }
    return next(error);
  });
}
const chapterImportHandlers = createChapterImportHandlers();

function isMissingModelRegistrySchemaError(error) {
  return error?.code === 'PGRST205' || /model_versions/i.test(error?.message || '');
}

function modelRegistrySchemaErrorResponse(res, error) {
  if (isMissingModelRegistrySchemaError(error)) {
    return res.status(503).json({
      error: "Registre de versions de modèles non installé.",
      details: "La table model_versions est indisponible dans Supabase.",
    });
  }
  return null;
}

router.get('/mangas/all', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { data, error } = await supabaseAdmin.from('mangas').select('*').order('titre');
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

router.patch('/mangas/:id/toggle', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { id } = req.params;
  const { data: manga, error: fetchErr } = await supabaseAdmin.from('mangas').select('enabled').eq('id', id).single();
  if (fetchErr || !manga) return res.status(404).json({ error: "Manga introuvable." });

  const { data, error } = await supabaseAdmin.from('mangas').update({ enabled: !manga.enabled }).eq('id', id).select().single();
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

router.post('/tomes', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { numero, titre } = req.body;
  let { manga } = req.query;
  if (Array.isArray(manga)) manga = manga[0];

  if (!numero || !titre) return res.status(400).json({ error: "Requis: numero, titre" });
  if (!manga) return res.status(400).json({ error: "Manga contexte requis." });

  try {
    const { data: mangaData, error: mangaError } = await supabaseAdmin
      .from('mangas')
      .select('id')
      .eq('slug', manga)
      .single();

    if (mangaError || !mangaData) return res.status(404).json({ error: "Manga introuvable." });

    const { data, error } = await supabaseAdmin
      .from('tomes')
      .insert({
        numero: parseInt(numero),
        titre,
        manga_id: mangaData.id
      })
      .select()
      .single();

    if (error) throw error;
    res.status(201).json(data);
  } catch (error) {
    if (error.code === '23505') return res.status(409).json({ error: `Le tome ${numero} existe déjà pour ce manga.` });
    console.error("Erreur création tome:", error);
    res.status(500).json({ error: "Erreur serveur." });
  }
});

router.post('/chapitres/upload', authMiddleware, roleCheck(['Admin']), chapterArchiveUploadMiddleware, validateRequest(
  { body: chapterUploadBodySchema },
  { onInvalid: (req) => req.file?.path && fs.existsSync(req.file.path) && fs.unlinkSync(req.file.path) }
), chapterImportHandlers.create);

router.get('/chapter-imports/:id', authMiddleware, roleCheck(['Admin']), chapterImportHandlers.get);

router.get('/hierarchy', authMiddleware, roleCheck(['Admin', 'Modo']), async (req, res) => {
  try {
    const { manga } = req.query; // Get manga slug

    let query = supabaseAdmin
      .from('tomes')
      .select(`
        id, numero, titre,
        mangas!inner(slug),
        chapitres (
          id, numero, titre,
          pages (
            id, numero_page, statut, url_image,
            bulles ( count )
          )
        )
      `)
      .order('numero', { ascending: true });

    if (manga) {
      query = query.eq('mangas.slug', manga);
    }

    const { data, error } = await query;

    if (error) throw error;

    // Sort nested manually if needed, or rely on client. Supabase nested order is tricky sometimes.
    // Let's sort chapters and pages in JS to be safe
    data.forEach(tome => {
      tome.chapitres.sort((a, b) => a.numero - b.numero);
      tome.chapitres.forEach(chap => {
        chap.pages.sort((a, b) => a.numero_page - b.numero_page);
        chap.pages.forEach(page => {
          page.url_image = getPageImagePath(page.id);
        });
      });
    });

    res.status(200).json(data);
  } catch (error) {
    console.error("Erreur hiérarchie:", error);
    res.status(500).json({ error: "Erreur lors de la récupération des données." });
  }
});

router.get('/pages/:id/bulles', authMiddleware, roleCheck(['Admin', 'Modo']), async (req, res) => {
  const { id } = req.params;
  try {
    const { data, error } = await supabaseAdmin
      .from('bulles')
      .select('id, x, y, w, h, texte_propose, statut, id_user_createur, order')
      .eq('id_page', id)
      .order('order', { ascending: true });

    if (error) throw error;
    res.status(200).json(data);
  } catch (error) {
    console.error("Erreur bulles admin:", error);
    res.status(500).json({ error: "Erreur lors de la récupération des bulles." });
  }
});

router.get('/banned-ips', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  try {
    const { data, error } = await supabaseAdmin
      .from('banned_ips')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) throw error;
    res.status(200).json(data);
  } catch (error) {
    console.error("Erreur banned_ips:", error);
    res.status(500).json({ error: "Erreur récupération IPs." });
  }
});

router.post('/banned-ips', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { ip, reason } = req.body;
  if (!ip) return res.status(400).json({ error: "IP requise" });

  try {
    const { data, error } = await supabaseAdmin
      .from('banned_ips')
      .insert({ ip, reason })
      .select()
      .single();

    if (error) {
      if (error.code === '23505') return res.status(409).json({ error: "Cette IP est déjà bannie." });
      throw error;
    }
    res.status(201).json(data);
  } catch (error) {
    console.error("Erreur ban IP:", error);
    res.status(500).json({ error: "Erreur lors du bannissement." });
  }
});

router.delete('/banned-ips/:ip', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { ip } = req.params;
  try {
    const { error } = await supabaseAdmin
      .from('banned_ips')
      .delete()
      .eq('ip', ip);

    if (error) throw error;
    res.status(200).json({ message: "IP débannie" });
  } catch (error) {
    console.error("Erreur deban IP:", error);
    res.status(500).json({ error: "Erreur lors du débannissement." });
  }
});

router.get('/covers', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  try {
    const { manga } = req.query;
    if (!manga) return res.status(400).json({ error: "Manga requis." });

    const { data: tomes, error } = await supabaseAdmin
      .from('tomes')
      .select('id, numero, titre, cover_url, mangas!inner(slug)')
      .eq('mangas.slug', manga)
      .order('numero', { ascending: true });

    if (error) throw error;

    const { data: mangaData, error: mangaError } = await supabaseAdmin
      .from('mangas')
      .select('id, titre, cover_url')
      .eq('slug', manga)
      .single();

    if (mangaError) throw mangaError;

    res.status(200).json({
      manga: mangaData,
      tomes: tomes
    });
  } catch (error) {
    console.error("Erreur covers:", error);
    res.status(500).json({ error: "Erreur lors de la récupération des couvertures." });
  }
});

router.post('/covers', authMiddleware, roleCheck(['Admin']), upload.single('cover'), async (req, res) => {
  const { type, id } = req.body;
  const file = req.file;

  if (!type || !id || !file) {
    if (file && fs.existsSync(file.path)) fs.unlinkSync(file.path);
    return res.status(400).json({ error: "Type, id et fichier sont requis." });
  }

  try {
    const fileBuffer = fs.readFileSync(file.path);
    const extension = path.extname(file.originalname);
    const safeFileName = `${type}-${id}-${Date.now()}${extension}`;
    const storagePath = `covers/${safeFileName}`;
    const contentType = mime.lookup(extension) || 'image/jpeg';

    await s3Client.send(new PutObjectCommand({
      Bucket: BUCKET_NAME,
      Key: storagePath,
      Body: fileBuffer,
      ContentType: contentType,
      CacheControl: 'public, max-age=31536000',
    }));

    const publicUrl = `${PUBLIC_URL_BASE}/${storagePath}`;

    if (type === 'manga') {
      const { error } = await supabaseAdmin
        .from('mangas')
        .update({ cover_url: publicUrl })
        .eq('id', id);
      if (error) throw error;
    } else if (type === 'tome') {
      const { error } = await supabaseAdmin
        .from('tomes')
        .update({ cover_url: publicUrl })
        .eq('id', id);
      if (error) throw error;
    } else {
      throw new Error("Type invalide.");
    }

    res.status(200).json({ url: publicUrl });
  } catch (error) {
    console.error("Erreur upload cover:", error);
    res.status(500).json({ error: "Erreur lors de l'upload de la couverture." });
  } finally {
    if (file && fs.existsSync(file.path)) fs.unlinkSync(file.path);
  }
});

const AI_MODEL_KEYS = [
  'model_ocr',
  'model_description',
  'model_chatgpt_ocr',
  'gemini_thinking_level',
  'chatgpt_reasoning_effort',
  'chatgpt_fast_mode',
];
const DEFAULT_MODELS = {
  model_ocr: 'gemini-2.5-flash-lite',
  model_description: 'gemini-3-flash-preview',
  model_chatgpt_ocr: 'gpt-5.6-luna',
  gemini_thinking_level: 'default',
  chatgpt_reasoning_effort: 'low',
  chatgpt_fast_mode: false,
};
const AI_MODEL_ID_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,99}$/;
const GEMINI_THINKING_LEVELS = new Set(['default', 'none', 'minimal', 'low', 'medium', 'high']);
const OPENAI_REASONING_EFFORTS = new Set(['default', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']);

let aiModelsCache = null;
let aiModelsCacheTime = 0;
const CACHE_TTL = 60 * 1000;

async function getAiModelsFromDb() {
  const now = Date.now();
  if (aiModelsCache && (now - aiModelsCacheTime) < CACHE_TTL) {
    return aiModelsCache;
  }

  const { data, error } = await supabaseAdmin
    .from('app_settings')
    .select('key, value')
    .in('key', AI_MODEL_KEYS);

  if (error) throw error;

  const models = { ...DEFAULT_MODELS };
  (data || []).forEach(row => {
    models[row.key] = row.key === 'chatgpt_fast_mode' ? row.value === 'true' : row.value;
  });

  aiModelsCache = models;
  aiModelsCacheTime = now;
  return models;
}

router.get('/ai-models', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  try {
    const models = await getAiModelsFromDb();
    res.json(models);
  } catch (error) {
    console.error("Erreur get AI models:", error);
    res.status(500).json({ error: "Erreur récupération des modèles IA." });
  }
});

router.put('/ai-models', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const {
    model_ocr,
    model_description,
    model_chatgpt_ocr,
    gemini_thinking_level,
    chatgpt_reasoning_effort,
    chatgpt_fast_mode,
  } = req.body;

  if (![model_ocr, model_description, model_chatgpt_ocr].every(value => typeof value === 'string' && AI_MODEL_ID_PATTERN.test(value.trim()))
      || !GEMINI_THINKING_LEVELS.has(gemini_thinking_level)
      || !OPENAI_REASONING_EFFORTS.has(chatgpt_reasoning_effort)
      || typeof chatgpt_fast_mode !== 'boolean') {
    return res.status(400).json({ error: 'Configuration IA invalide.' });
  }

  try {
    const updates = [
      { key: 'model_ocr', value: model_ocr.trim() },
      { key: 'model_description', value: model_description.trim() },
      { key: 'model_chatgpt_ocr', value: model_chatgpt_ocr.trim() },
      { key: 'gemini_thinking_level', value: gemini_thinking_level },
      { key: 'chatgpt_reasoning_effort', value: chatgpt_reasoning_effort },
      { key: 'chatgpt_fast_mode', value: String(chatgpt_fast_mode) },
    ];

    for (const { key, value } of updates) {
      const { error } = await supabaseAdmin
        .from('app_settings')
        .upsert({ key, value }, { onConflict: 'key' });
      if (error) throw error;
    }

    aiModelsCache = null;
    const models = await getAiModelsFromDb();
    res.json(models);
  } catch (error) {
    console.error("Erreur update AI models:", error);
    res.status(500).json({ error: "Erreur mise à jour des modèles IA." });
  }
});

router.get('/ai-models/public', async (req, res) => {
  try {
    const models = await getAiModelsFromDb();
    res.json(models);
  } catch (error) {
    console.error("Erreur get public AI models:", error);
    res.status(500).json({ error: "Erreur récupération des modèles IA." });
  }
});

router.get('/prompts', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  try {
    const registry = await getPromptRegistry();
    const prompts = [...registry.values()]
      .map(({ content, ...prompt }) => ({ ...prompt, content }));
    res.json(prompts);
  } catch (error) {
    console.error("Erreur get prompts:", error);
    res.status(500).json({ error: "Erreur récupération des prompts." });
  }
});

router.put('/prompts', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { prompts } = req.body;

  if (!prompts || typeof prompts !== 'object' || Array.isArray(prompts)) {
    return res.status(400).json({ error: "Corps de requête invalide : { prompts: { [key]: content } } attendu." });
  }

  const entries = Object.entries(prompts)
    .map(([key, content]) => ({ key, content: typeof content === 'string' ? content.trim() : '' }));

  if (entries.length === 0) {
    return res.status(400).json({ error: "Aucun prompt à mettre à jour." });
  }

  const invalid = entries.find(({ key, content }) => (
    !PROMPT_KEYS.has(key)
    || content.length === 0
    || content.length > PROMPT_CONTENT_MAX_LENGTH
  ));
  if (invalid) {
    return res.status(400).json({
      error: `Prompt invalide : « ${invalid.key} » (clé inconnue ou contenu vide / > ${PROMPT_CONTENT_MAX_LENGTH} caractères).`,
    });
  }

  try {
    for (const { key, content } of entries) {
      const fallback = getDefaultPrompt(key);
      const query = content === fallback.content
        ? supabaseAdmin.from('llm_prompts').delete().eq('key', key)
        : supabaseAdmin.from('llm_prompts').upsert({
          key,
          label: fallback.label,
          category: fallback.category,
          description: fallback.description,
          content,
          updated_by: req.user?.id ?? null,
        }, { onConflict: 'key' });
      const { error } = await query;
      if (error) throw error;
    }

    invalidatePromptCache();
    const registry = await getPromptRegistry();
    const updated = [...registry.values()]
      .map(({ content, ...prompt }) => ({ ...prompt, content }));
    res.json(updated);
  } catch (error) {
    console.error("Erreur update prompts:", error);
    res.status(500).json({ error: "Erreur mise à jour des prompts." });
  }
});

router.get('/prompts/public', async (req, res) => {
  try {
    res.json(await getPromptContents());
  } catch (error) {
    console.error("Erreur get public prompts:", error);
    res.status(500).json({ error: "Erreur récupération des prompts." });
  }
});


router.post('/upload/page', authMiddleware, roleCheck(['Admin']), uploadSinglePage, async (req, res) => {
  const { key } = req.body;
  const file = req.file;

  if (!key || !file) {
    if (file && fs.existsSync(file.path)) fs.unlinkSync(file.path);
    return res.status(400).json({ error: "key et file sont requis." });
  }

  try {
    const normalizedKey = normalizePageStorageKey(key);
    const { buffer: fileBuffer, contentType } = await preparePageUpload(file.path);

    const pagesBucketName = getPrivatePagesBucketName();
    await s3Client.send(new PutObjectCommand({
      Bucket: pagesBucketName,
      Key: normalizedKey,
      Body: fileBuffer,
      ContentType: contentType,
      CacheControl: 'private, no-store',
    }));
    clearBubbleGeometryCache();

    const pageStorageRef = createPageStorageRef(pagesBucketName, normalizedKey);
    res.json({ url: pageStorageRef });
  } catch (error) {
    if (error instanceof PageStorageError && error.code === 'INVALID_PAGE_KEY') {
      return res.status(400).json({ error: 'La clé de stockage de la page est invalide.' });
    }
    if (error?.code === 'PAGE_IMAGE_TOO_LARGE') {
      return res.status(413).json({ error: 'L’image de page dépasse la taille maximale autorisée.' });
    }
    if (isPageImageValidationError(error)) {
      return res.status(415).json({ error: "Le fichier doit être une image JPEG, PNG, WebP ou AVIF valide." });
    }
    console.error("Erreur upload page:", error);
    return res.status(500).json({ error: "Erreur upload vers R2." });
  } finally {
    if (file && fs.existsSync(file.path)) fs.unlinkSync(file.path);
  }
});

router.post('/tomes/batch-pages', authMiddleware, roleCheck(['Admin']), express.json({ limit: '10mb' }), async (req, res) => {
  const { tome_id, chapters } = req.body;

  if (!tome_id || !chapters || !Array.isArray(chapters) || chapters.length === 0) {
    return res.status(400).json({ error: "tome_id et chapters sont requis." });
  }

  try {
    const results = [];

    for (const chapter of chapters) {
      const { data: newChap, error: chapError } = await supabaseAdmin
        .from('chapitres')
        .insert({ id_tome: tome_id, numero: parseInt(chapter.numero), titre: chapter.titre })
        .select()
        .single();

      if (chapError) {
        if (chapError.code === '23505') {
          results.push({ numero: chapter.numero, error: `Le chapitre ${chapter.numero} existe déjà.` });
          continue;
        }
        throw chapError;
      }

      const pagesToInsert = chapter.pages.map(p => ({
        id_chapitre: newChap.id,
        numero_page: p.numero_page,
        url_image: p.url_image,
        statut: 'not_started'
      }));

      const { error: pagesError } = await supabaseAdmin
        .from('pages')
        .insert(pagesToInsert);

      if (pagesError) throw pagesError;

      results.push({ numero: chapter.numero, id: newChap.id, pages: pagesToInsert.length });
    }

    res.status(201).json({ message: "Batch créé avec succès.", results });
  } catch (error) {
    console.error("Erreur batch-pages:", error);
    res.status(500).json({ error: "Erreur lors de la création batch.", details: error.message });
  }
});

const EMBEDDING_STATS_PAGE_SIZE = 1000;
const MAX_EMBEDDING_STATS_ROWS = 50000;

async function getAllAiEmbeddingStats(mangaSlug) {
  const rows = [];
  for (let offset = 0; offset < MAX_EMBEDDING_STATS_ROWS; offset += EMBEDDING_STATS_PAGE_SIZE) {
    const { data, error } = await supabaseAdmin
      .rpc('get_ai_embedding_stats', { p_manga_slug: mangaSlug || null })
      .range(offset, offset + EMBEDDING_STATS_PAGE_SIZE - 1);
    if (error) throw error;

    const batch = data || [];
    rows.push(...batch);
    if (batch.length < EMBEDDING_STATS_PAGE_SIZE) return rows;
  }

  throw new Error(`Le nombre de pages dépasse la limite de ${MAX_EMBEDDING_STATS_ROWS}.`);
}

router.get('/ai-models/embedding-stats', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  try {
    const { manga } = req.query;
    const data = await getAllAiEmbeddingStats(manga);

    const stats = data.map(page => ({
      id: page.id,
      chapitre_id: page.chapitre_id,
      chapitre_numero: page.chapitre_numero,
      tome_numero: page.tome_numero,
      numero: page.numero,
      url_image: getPageImagePath(page.id),
      description: page.description,
      has_voyage: page.has_voyage,
      has_gemini: page.has_gemini,
      has_f2llm: page.has_f2llm,
      has_description: page.has_description,
    }));

    res.json(stats);
  } catch (error) {
    console.error("Erreur embedding-stats:", error);
    res.status(500).json({ error: "Erreur lors de la récupération des statistiques d'embeddings." });
  }
});


router.post('/ai-models/save-page-data', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { id_page, description, embedding_voyage, embedding_gemini, embedding_f2llm } = req.body;

  if (!id_page) return res.status(400).json({ error: "Page ID requis." });

  try {
    const updateData = {};
    if (description) updateData.description = typeof description === 'string' ? description : JSON.stringify(description);
    if (embedding_voyage) updateData.embedding_voyage = embedding_voyage;
    if (embedding_gemini) updateData.embedding_gemini = embedding_gemini;
    if (embedding_f2llm) updateData.embedding_f2llm = embedding_f2llm;

    const { error } = await supabaseAdmin
      .from('pages')
      .update(updateData)
      .eq('id', id_page);

    if (error) throw error;
    res.json({ success: true });
  } catch (error) {
    console.error("Erreur save-page-data:", error);
    res.status(500).json({ error: "Erreur lors de la sauvegarde des données de la page." });
  }
});

router.post('/ai-models/generate-voyage-embedding', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: "Texte requis." });

  try {
    const embedding = await generateVoyageEmbedding(text, "document");
    res.json({ embedding });
  } catch (error) {
    console.error("Erreur generation Voyage:", error);
    res.status(500).json({ error: "Erreur lors de la génération de l'embedding Voyage." });
  }
});

router.post('/ai-models/generate-f2llm-embedding', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: "Texte requis." });

  try {
    const embedding = await generateF2llmEmbedding(text, "document");
    res.json({ embedding });
  } catch (error) {
    console.error("Erreur generation F2LLM:", error);
    res.status(500).json({ error: error.message || "Erreur lors de la generation de l'embedding F2LLM." });
  }
});

router.post('/ai-models/trigger-backfill', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { manga } = req.query;
  res.json({ message: "Processus de backfill Gemini (multimodal) démarré en tâche de fond." });

  (async () => {
    try {
      console.log(`[Backfill] Démarrage du backfill Gemini multimodal${manga ? ` pour ${manga}` : ''}...`);

      let query = supabaseAdmin
        .from('pages')
        .select(`
            id, description, url_image,
            bulles ( texte_propose, statut ),
            chapitres!inner( tomes!inner( mangas!inner(slug) ) )
        `)
        .is('embedding_gemini', null);

      if (manga) {
        query = query.eq('chapitres.tomes.mangas.slug', manga);
      }

      const { data: pagesToProcess, error: pagesError } = await query;
      if (pagesError) throw pagesError;

      let processed = 0;
      let errors = 0;

      for (const page of pagesToProcess) {
        let contentToEmbed = "";

        if (page.description) {
          let desc = page.description;
          try {
            if (typeof desc === 'string') desc = JSON.parse(desc).content;
            else if (typeof desc === 'object') desc = desc.content;
          } catch (e) { }
          if (desc) contentToEmbed += desc + " ";
        }

        if (page.bulles && page.bulles.length > 0) {
          const texts = page.bulles
            .filter(b => b.statut === 'validated' && b.texte_propose)
            .map(b => b.texte_propose)
            .join(' ');
          if (texts) contentToEmbed += texts;
        }

        contentToEmbed = contentToEmbed.trim();

        if (contentToEmbed.length > 0) {
          try {
            const embedding = await generateGeminiEmbedding(contentToEmbed, "RETRIEVAL_DOCUMENT", page.url_image || null);

            const { error: updateError } = await supabaseAdmin
              .from('pages')
              .update({ embedding_gemini: embedding })
              .eq('id', page.id);

            if (updateError) {
              console.error(`[Backfill] Erreur Update Supabase pour la page ${page.id}:`, updateError);
              errors++;
            } else {
              processed++;
            }
          } catch (embedError) {
            console.error(`[Backfill] Erreur Gemini Embedding pour la page ${page.id}:`, embedError.message);
            errors++;
          }
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }

      console.log(`[Backfill] Terminé. Traitées: ${processed}, Erreurs: ${errors}`);
    } catch (e) {
      console.error("[Backfill] Erreur globale lors du backfill:", e);
    }
  })();
});

router.post('/ai-models/trigger-backfill-voyage', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { manga } = req.query;
  res.json({ message: "Processus de backfill Voyage démarré en tâche de fond." });

  (async () => {
    try {
      console.log(`[Backfill Voyage] Démarrage du backfill Voyage${manga ? ` pour ${manga}` : ''}...`);

      let query = supabaseAdmin
        .from('pages')
        .select(`
            id, description,
            bulles ( texte_propose, statut ),
            chapitres!inner( tomes!inner( mangas!inner(slug) ) )
        `)
        .is('embedding_voyage', null);

      if (manga) {
        query = query.eq('chapitres.tomes.mangas.slug', manga);
      }

      const { data: pagesToProcess, error: pagesError } = await query;
      if (pagesError) throw pagesError;

      let processed = 0;
      let errors = 0;

      for (const page of pagesToProcess) {
        let contentToEmbed = "";

        if (page.description) {
          let desc = page.description;
          try {
            if (typeof desc === 'string') desc = JSON.parse(desc).content;
            else if (typeof desc === 'object') desc = desc.content;
          } catch (e) { }
          if (desc) contentToEmbed += desc + " ";
        }

        if (page.bulles && page.bulles.length > 0) {
          const texts = page.bulles
            .filter(b => b.statut === 'validated' && b.texte_propose)
            .map(b => b.texte_propose)
            .join(' ');
          if (texts) contentToEmbed += texts;
        }

        contentToEmbed = contentToEmbed.trim();

        if (contentToEmbed.length > 0) {
          try {
            const embedding = await generateVoyageEmbedding(contentToEmbed, "document");

            const { error: updateError } = await supabaseAdmin
              .from('pages')
              .update({ embedding_voyage: embedding })
              .eq('id', page.id);

            if (updateError) {
              console.error(`[Backfill Voyage] Erreur Update Supabase pour la page ${page.id}:`, updateError);
              errors++;
            } else {
              processed++;
            }
          } catch (embedError) {
            console.error(`[Backfill Voyage] Erreur Voyage Embedding pour la page ${page.id}:`, embedError.message);
            errors++;
          }
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      }

      console.log(`[Backfill Voyage] Terminé. Traitées: ${processed}, Erreurs: ${errors}`);
    } catch (e) {
      console.error("[Backfill Voyage] Erreur globale lors du backfill:", e);
    }
  })();
});

router.post('/ai-models/trigger-backfill-f2llm', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  const { manga } = req.query;
  res.json({ message: "Processus de backfill F2LLM demarre en tache de fond." });

  (async () => {
    try {
      console.log(`[Backfill F2LLM] Demarrage${manga ? ` pour ${manga}` : ''}...`);

      let query = supabaseAdmin
        .from('pages')
        .select(`
            id, description,
            bulles ( texte_propose, statut ),
            chapitres!inner( tomes!inner( mangas!inner(slug) ) )
        `)
        .is('embedding_f2llm', null)
        .not('description', 'is', null);

      if (manga) {
        query = query.eq('chapitres.tomes.mangas.slug', manga);
      }

      const { data: pagesToProcess, error: pagesError } = await query;
      if (pagesError) throw pagesError;

      let processed = 0;
      let skipped = 0;
      let errors = 0;

      for (const page of pagesToProcess || []) {
        const contentToEmbed = buildPageEmbeddingText(page);
        if (!contentToEmbed) {
          skipped++;
          continue;
        }

        try {
          const embedding = await generateF2llmEmbedding(contentToEmbed, "document");
          const { error: updateError } = await supabaseAdmin
            .from('pages')
            .update({ embedding_f2llm: embedding })
            .eq('id', page.id);

          if (updateError) throw updateError;
          processed++;
        } catch (embedError) {
          console.error(`[Backfill F2LLM] Erreur pour la page ${page.id}:`, embedError.message);
          errors++;
        }
      }

      console.log(`[Backfill F2LLM] Termine. Traitees: ${processed}, ignorees: ${skipped}, erreurs: ${errors}`);
    } catch (e) {
      console.error("[Backfill F2LLM] Erreur globale:", e);
    }
  })();
});

router.post('/model-versions/:id/promote', authMiddleware, roleCheck(['Admin']), async (req, res) => {
  try {
    const { data: version, error } = await supabaseAdmin
      .from('model_versions')
      .select('*')
      .eq('id', req.params.id)
      .single();

    if (error) {
      const schemaResponse = modelRegistrySchemaErrorResponse(res, error);
      if (schemaResponse) return schemaResponse;
      return res.status(404).json({ error: "Version modèle introuvable." });
    }
    if (!version) return res.status(404).json({ error: "Version modèle introuvable." });

    await supabaseAdmin
      .from('model_versions')
      .update({ is_active: false })
      .eq('kind', version.kind)
      .eq('is_active', true);

    const promotedAt = new Date().toISOString();
    const { data: promoted, error: promoteError } = await supabaseAdmin
      .from('model_versions')
      .update({
        is_candidate: false,
        is_active: true,
        promoted_at: promotedAt,
      })
      .eq('id', version.id)
      .select()
      .single();

    if (promoteError) throw promoteError;

    await supabaseAdmin
      .from('app_settings')
      .upsert({
        key: `active_model_version_${version.kind}`,
        value: JSON.stringify({
          model_version_id: promoted.id,
          kind: promoted.kind,
          hf_repo: promoted.hf_repo,
          hf_revision: promoted.hf_revision,
          promoted_at: promotedAt,
        }),
      }, { onConflict: 'key' });

    res.json(promoted);
  } catch (error) {
    console.error("Erreur promotion model version:", error);
    const schemaResponse = modelRegistrySchemaErrorResponse(res, error);
    if (schemaResponse) return schemaResponse;
    res.status(500).json({ error: "Erreur promotion du modèle.", details: error.message });
  }
});


module.exports = router;


