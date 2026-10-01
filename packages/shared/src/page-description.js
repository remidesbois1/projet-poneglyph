const ASTRA_DESCRIPTION_MODEL = 'gpt-6-astra';
const DESCRIPTION_PROMPT_VERSION = 2;
const ASTRA_REASONING_EFFORT = 'high';

function hasExactKeys(value, keys) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value).length === keys.length
    && keys.every((key) => Object.prototype.hasOwnProperty.call(value, key));
}

// Deliberately no fence stripping, JSON repair, coercion or inferred metadata.
function parsePageDescription(value) {
  let description = value;
  try {
    if (typeof value === 'string') description = JSON.parse(value);
  } catch {
    throw new Error('Description invalide : JSON strict attendu.');
  }
  if (!hasExactKeys(description, ['content', 'metadata'])
    || typeof description.content !== 'string'
    || !description.content.trim() || description.content.length > 20000
    || !hasExactKeys(description.metadata, ['arc', 'characters'])
    || typeof description.metadata.arc !== 'string' || description.metadata.arc.length > 300
    || !Array.isArray(description.metadata.characters) || description.metadata.characters.length > 100
    || !description.metadata.characters.every((name) => typeof name === 'string' && name.trim() && name.length <= 200)) {
    throw new Error('Description invalide : content, metadata.arc et metadata.characters sont requis.');
  }
  return {
    content: description.content.trim(),
    metadata: {
      arc: description.metadata.arc.trim(),
      characters: [...new Set(description.metadata.characters.map((name) => name.trim()))],
    },
  };
}

function isCurrentAstraPage(page) {
  if (page.description_model !== ASTRA_DESCRIPTION_MODEL
    || page.description_prompt_version !== DESCRIPTION_PROMPT_VERSION
    || !page.has_description || !page.has_voyage || !page.has_gemini) return false;
  try {
    parsePageDescription(page.description);
    return true;
  } catch {
    return false;
  }
}

module.exports = { ASTRA_DESCRIPTION_MODEL, DESCRIPTION_PROMPT_VERSION, ASTRA_REASONING_EFFORT, parsePageDescription, isCurrentAstraPage };
