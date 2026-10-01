begin;

alter table public.pages
  add column if not exists description_model text,
  add column if not exists description_prompt_version integer,
  add column if not exists description_generated_at timestamptz;

-- A later legacy description/vector update must not keep an obsolete Astra completion marker.
create or replace function public.invalidate_page_description_provenance()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  if new.url_image is distinct from old.url_image
    or ((new.description is distinct from old.description
      or new.embedding_voyage is distinct from old.embedding_voyage
      or new.embedding_gemini is distinct from old.embedding_gemini)
      and (new.description_generated_at is not distinct from old.description_generated_at
        or new.description is null or new.embedding_voyage is null
        or new.embedding_gemini is null)) then
    new.description_model := null;
    new.description_prompt_version := null;
    new.description_generated_at := null;
  end if;
  return new;
end;
$$;

drop trigger if exists invalidate_page_description_provenance on public.pages;
create trigger invalidate_page_description_provenance
before update on public.pages
for each row execute function public.invalidate_page_description_provenance();

-- The RPC returns flags/provenance, never vectors or private storage references.
drop function if exists public.get_ai_embedding_stats(text);
create function public.get_ai_embedding_stats(p_manga_slug text default null)
returns table (
  id bigint, chapitre_id bigint, chapitre_numero integer, tome_numero integer,
  numero integer, description jsonb,
  has_voyage boolean, has_gemini boolean, has_f2llm boolean, has_description boolean,
  description_model text, description_prompt_version integer, description_generated_at timestamptz
)
language sql stable security definer
set search_path = pg_catalog, public
as $$
  select p.id, p.id_chapitre, c.numero, t.numero, p.numero_page, p.description,
    p.embedding_voyage is not null, p.embedding_gemini is not null, p.embedding_f2llm is not null,
    p.description is not null and p.description not in ('null'::jsonb, '""'::jsonb, '{}'::jsonb),
    p.description_model, p.description_prompt_version, p.description_generated_at
  from public.pages p
  join public.chapitres c on c.id = p.id_chapitre
  join public.tomes t on t.id = c.id_tome
  join public.mangas m on m.id = t.manga_id
  where p_manga_slug is null or m.slug = p_manga_slug
  order by t.numero, c.numero, p.numero_page, p.id;
$$;
revoke all on function public.get_ai_embedding_stats(text) from public, anon, authenticated;
grant execute on function public.get_ai_embedding_stats(text) to service_role;

-- Replace an existing override too; otherwise it would hide the new bundled prompt.
update public.llm_prompts
set content = $page_description_v2$
Analyse cette page de One Piece afin de produire une description sémantique destinée à la recherche vectorielle.

L'objectif est qu'une recherche formulée naturellement par un utilisateur retrouve cette page à partir de ce qui s'y passe réellement : personnages présents, actions, interactions, objets, lieu, événement, techniques et éléments importants visibles.

Réponds uniquement avec ce JSON strict :

{
  "content": "Description sémantique de la page.",
  "metadata": {
    "arc": "Nom de l'arc ou chaîne vide",
    "characters": ["Personnage 1", "Personnage 2"]
  }
}

Règles pour "content" :

- Décris en priorité les événements et actions visibles sur la page.
- Si la page contient plusieurs cases ou plusieurs actions importantes, mentionne-les toutes de manière concise, idéalement dans leur ordre narratif.
- Commence par l'événement ou l'action la plus distinctive de la page, puis ajoute les autres informations utiles.
- Utilise des phrases courtes, directes et factuelles : personnage + action + cible/contexte.
- Mentionne explicitement les interactions entre personnages : attaque, discussion, poursuite, réaction, protection, confrontation, observation, etc.
- Mentionne les objets, lieux, transformations, pouvoirs, attaques ou éléments de lore uniquement lorsqu'ils sont clairement visibles ou fortement établis par la page.
- Utilise les noms canoniques de One Piece lorsque l'identité est suffisamment certaine.
- Utilise des termes susceptibles d'être recherchés par un lecteur : noms des personnages, attaques, pouvoirs, lieux, objets importants et événements précis.
- Décris également une expression, réaction ou situation visuelle distinctive lorsqu'elle peut aider à retrouver spécifiquement cette page.
- Ne te limite pas au texte des bulles : exploite toute l'information visuelle de la page.
- Ne déduis pas un événement absent de l'image à partir de ta connaissance de One Piece.
- Ne complète pas une scène avec ce qui se passe avant ou après dans l'histoire.
- Ne nomme pas un personnage si son identité n'est pas suffisamment certaine visuellement.
- Ne nomme pas une attaque, une transformation, un lieu ou un pouvoir simplement parce qu'ils semblent plausibles.
- En cas de doute, utilise une description visuelle générique plutôt qu'une identification incertaine.
- N'invente jamais de dialogue, d'intention, de relation causale ou d'émotion qui ne soit pas clairement perceptible.
- Ne mentionne pas le style graphique, les traits, les hachures, la composition ou le cadrage sauf nécessité sémantique.

Règles pour "metadata" :

- "characters" contient uniquement les personnages identifiés avec une confiance élevée et réellement présents.
- N'ajoute pas un personnage simplement parce qu'il est mentionné dans un dialogue.
- "arc" contient le nom canonique uniquement s'il peut être déterminé avec une confiance élevée.
- Sinon `"arc": ""`.

La description doit être suffisamment riche pour distinguer cette page de pages proches, mais rester compacte. Privilégie les faits discriminants.

Ne produis aucune explication, aucun Markdown et aucun texte en dehors du JSON.
$page_description_v2$, updated_at = now()
where key = 'page_description';

commit;
