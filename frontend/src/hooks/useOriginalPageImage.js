import { useEffect, useState, useCallback } from "react";
import { useAuth } from "@/context/AuthContext";
import { fetchOriginalPageImage } from "@/lib/pageImageClient";

export function useOriginalPageImage(
  pageId,
  { thumbnail = false, width = 480 } = {},
) {
  const { session } = useAuth();
  const token = session?.access_token;
  const userId = session?.user?.id;
  const [result, setResult] = useState(null);
  const [generation, setGeneration] = useState(0);
  useEffect(() => {
    if (!token || !pageId) return;
    const controller = new AbortController();
    let url;
    fetchOriginalPageImage(pageId, token, {
      signal: controller.signal,
      thumbnail,
      width,
      expectedUserId: userId,
    })
      .then((blob) => {
        if (controller.signal.aborted) return;
        url = URL.createObjectURL(blob);
        setResult({ pageId, token, generation, thumbnail, width, url, error: null });
      })
      .catch((error) => {
        if (!controller.signal.aborted)
          setResult({
            pageId,
            token,
            generation,
            thumbnail,
            width,
            url: null,
            error: error.message,
          });
      });
    return () => {
      controller.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [pageId, token, userId, thumbnail, width, generation]);
  const current =
    result?.pageId === pageId &&
    result?.token === token &&
    result?.thumbnail === thumbnail &&
    result?.width === width &&
    result?.generation === generation
      ? result
      : null;
  const retry = useCallback(() => setGeneration((value) => value + 1), []);
  return {
    url: current?.url ?? null,
    error: current?.error ?? null,
    loading: !current,
    retry,
  };
}
