const multer = require('multer');
const inputLimits = require('@poneglyph/shared/input-limits.json');

function createPageUploadMiddleware(storage) {
  const pageUpload = multer({
    storage,
    limits: {
      fileSize: inputLimits.pageImageBytes,
      files: 1,
      fields: 1,
      // Busboy emits partsLimit as soon as this count is reached.
      parts: 3,
      fieldNameSize: 100,
      fieldSize: 2048,
    },
  });

  return function uploadSinglePage(req, res, next) {
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
  };
}

module.exports = { createPageUploadMiddleware };
