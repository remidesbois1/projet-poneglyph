-- Migration pour la recherche par image
-- Exécutez ce script dans l'éditeur SQL de Supabase

-- 1. Ajouter la colonne image_embedding à la table pages
ALTER TABLE public.pages ADD COLUMN IF NOT EXISTS image_embedding vector(768);

-- 2. Créer l'index vectoriel pour améliorer les performances (HNSW)
-- On utilise cosine distance (vector_cosine_ops) pour les embeddings d'images
CREATE INDEX IF NOT EXISTS pages_image_embedding_idx ON public.pages USING hnsw (image_embedding vector_cosine_ops);

-- 3. Créer la fonction RPC match_pages_image
CREATE OR REPLACE FUNCTION match_pages_image (
  query_embedding vector(768),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id integer,
  manga_slug text,
  tome_numero integer,
  chapitre_numero float,
  numero_page integer,
  url_image text,
  description text,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    p.id,
    m.slug AS manga_slug,
    t.numero AS tome_numero,
    c.numero AS chapitre_numero,
    p.numero AS numero_page,
    p.url_image,
    p.description::text,
    1 - (p.image_embedding <=> query_embedding) AS similarity
  FROM pages p
  JOIN chapitres c ON p.chapitre_id = c.id
  JOIN tomes t ON c.tome_id = t.id
  JOIN mangas m ON t.manga_id = m.id
  WHERE 
    p.image_embedding IS NOT NULL
    AND 1 - (p.image_embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;
