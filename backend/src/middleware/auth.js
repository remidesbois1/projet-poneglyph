const { supabase, supabaseAdmin } = require('../config/supabaseClient');

function getBearerToken(req) {
  const authorization = req.headers.authorization;
  if (typeof authorization !== 'string') return null;

  const match = authorization.match(/^Bearer\s+(.+)$/i);
  return match?.[1]?.trim() || null;
}

function authUnavailable(res) {
  return res.status(503).json({
    code: 'SUPABASE_UNAVAILABLE',
    error: 'Vérification Supabase temporairement indisponible. Réessayez la reprise dans un instant.',
  });
}

function authRequired(res) {
  return res.status(401).json({ code: 'SUPABASE_AUTH_REQUIRED', error: 'Accès non autorisé : token manquant, invalide ou expiré.' });
}

function isTransientAuthError(error) {
  return error instanceof TypeError || error?.name === 'AuthRetryableFetchError'
    || error?.status === 0 || error?.status === 429 || error?.status >= 500
    || /fetch failed|failed to fetch|network|timeout|socket/i.test(error?.message || '');
}

function createAuthMiddleware({ authClient = supabase, profileClient = supabaseAdmin, optional = false } = {}) {
  return async (req, res, next) => {
    const token = getBearerToken(req);
    if (!token) return optional ? next() : authRequired(res);

    let result;
    try { result = await authClient.auth.getUser(token); }
    catch (error) { return isTransientAuthError(error) ? authUnavailable(res) : authRequired(res); }
    if (result.error) return isTransientAuthError(result.error) ? authUnavailable(res) : authRequired(res);
    const user = result.data?.user;
    if (!user) return authRequired(res);
    if (optional) {
      req.user = user;
      return next();
    }

    // Read the verified user's role without depending on anonymous profile policies.
    let profileResult;
    try {
      profileResult = await profileClient.from('profiles').select('role').eq('id', user.id).maybeSingle();
    } catch { return authUnavailable(res); }
    if (profileResult.error) return authUnavailable(res);
    if (!profileResult.data) {
      return res.status(403).json({ code: 'SUPABASE_PERMISSION_DENIED', error: 'Accès refusé : profil utilisateur introuvable.' });
    }
    req.user = { ...user, role: profileResult.data.role };
    return next();
  };
}

const authMiddleware = createAuthMiddleware();
const optionalAuthMiddleware = createAuthMiddleware({ optional: true });

const roleCheck = (allowedRoles) => {
  return (req, res, next) => {
    const userRole = req.user?.role;
    if (userRole && allowedRoles.includes(userRole)) {
      next();
    } else {
      res.status(403).json({ code: 'SUPABASE_PERMISSION_DENIED', error: 'Accès refusé : permissions insuffisantes.' });
    }
  };
};

module.exports = { authMiddleware, createAuthMiddleware, getBearerToken, optionalAuthMiddleware, roleCheck };
