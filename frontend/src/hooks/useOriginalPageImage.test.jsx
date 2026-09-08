import { renderHook, waitFor, act } from "@testing-library/react";
import { vi, it, expect, beforeEach, afterEach } from "vitest";
import { useOriginalPageImage } from "./useOriginalPageImage";
import { fetchOriginalPageImage } from "@/lib/pageImageClient";
vi.mock("@/context/AuthContext", () => ({
  useAuth: () => ({ session: { access_token: "test-session" } }),
}));
vi.mock("@/lib/pageImageClient", () => ({ fetchOriginalPageImage: vi.fn() }));
beforeEach(() => {
  vi.clearAllMocks();
  URL.createObjectURL = vi.fn(() => "blob:original");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.restoreAllMocks());
it("loads authenticated originals and releases the blob and request on unmount", async () => {
  fetchOriginalPageImage.mockResolvedValue(new Blob(["image"]));
  const { result, unmount } = renderHook(() => useOriginalPageImage(42));
  await waitFor(() => expect(result.current.url).toBe("blob:original"));
  expect(fetchOriginalPageImage).toHaveBeenCalledWith(
    42,
    "test-session",
    expect.objectContaining({
      thumbnail: false,
      signal: expect.any(AbortSignal),
    }),
  );
  const signal = fetchOriginalPageImage.mock.calls[0][2].signal;
  unmount();
  expect(signal.aborted).toBe(true);
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:original");
});
it("requests protected thumbnails and supports retrying failures", async () => {
  fetchOriginalPageImage
    .mockRejectedValueOnce(new Error("Network"))
    .mockResolvedValue(new Blob(["image"]));
  const { result } = renderHook(() =>
    useOriginalPageImage(42, { thumbnail: true, width: 480 }),
  );
  await waitFor(() => expect(result.current.error).toBe("Network"));
  act(() => result.current.retry());
  await waitFor(() => expect(result.current.url).toBe("blob:original"));
  expect(fetchOriginalPageImage).toHaveBeenLastCalledWith(
    42,
    "test-session",
    expect.objectContaining({ thumbnail: true, width: 480 }),
  );
});
it("ignores responses from a previous page", async () => {
  let resolveFirst;
  fetchOriginalPageImage
    .mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveFirst = resolve;
        }),
    )
    .mockResolvedValue(new Blob(["second"]));
  const { result, rerender } = renderHook(
    ({ id }) => useOriginalPageImage(id),
    { initialProps: { id: 1 } },
  );
  rerender({ id: 2 });
  await waitFor(() => expect(result.current.url).toBe("blob:original"));
  await act(async () => resolveFirst(new Blob(["stale"])));
  expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
});
