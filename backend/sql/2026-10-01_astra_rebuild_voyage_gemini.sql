-- Upgrade installations that already applied the Astra rebuild migration.
-- Completion now requires only Voyage and Gemini; F2LLM is independent.
begin;

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

commit;
