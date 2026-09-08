import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
import { vi, beforeEach, it, expect } from "vitest";
import PageReview from "./page";
const drag = vi.hoisted(() => ({ onEnd: null }));
vi.mock("@dnd-kit/core", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    DndContext: ({ children, onDragEnd }) => {
      drag.onEnd = onDragEnd;
      return children;
    },
  };
});
import { getPageById, getBubblesForPage, reorderBubbles } from "@/lib/api";
vi.mock("next/navigation", () => ({
  useParams: () => ({ pageId: "42" }),
  useRouter: () => ({ push: vi.fn() }),
}));
vi.mock("@/context/AuthContext", () => ({
  useAuth: () => ({ session: { access_token: "token" } }),
}));
vi.mock("@/context/MangaContext", () => ({
  useManga: () => ({ mangaSlug: "one-piece" }),
}));
vi.mock("@/lib/api", () => ({
  getPageById: vi.fn(),
  getBubblesForPage: vi.fn(),
  reorderBubbles: vi.fn(),
  approvePage: vi.fn(),
  rejectPage: vi.fn(),
  rejectBubble: vi.fn(),
  savePageDescription: vi.fn(),
  getMetadataSuggestions: vi.fn(),
}));
vi.mock("@/lib/geminiClient", () => ({ generatePageDescription: vi.fn() }));
vi.mock("@/components/moderation/ReviewPageImage", () => ({
  default: () => <div>Image originale</div>,
}));
vi.mock("@/components/ValidationForm", () => ({ default: () => null }));
vi.mock("@/components/AiAccessDialog", () => ({ default: () => null }));
vi.mock("@/components/ModerationCommentModal", () => ({ default: () => null }));
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }));
beforeEach(() => {
  vi.clearAllMocks();
  getPageById.mockResolvedValue({
    data: {
      id: 42,
      numero_page: 7,
      statut: "pending_review",
      url_image: "/page.png",
    },
  });
  getBubblesForPage.mockResolvedValue({
    data: [
      { id: 12, order: 2, texte_propose: "Deuxième" },
      { id: 11, order: 1, texte_propose: "Première" },
    ],
  });
  reorderBubbles.mockResolvedValue({ data: {} });
});
it("finishes loading after successful requests and displays reading order", async () => {
  render(<PageReview />);
  expect(await screen.findByText("Première")).toBeInTheDocument();
  expect(screen.queryByText("Chargement de la page…")).not.toBeInTheDocument();
  expect(screen.getAllByRole("listitem")[0]).toHaveTextContent("Première");
});
it("persists the complete contiguous order and disables page validation during saving", async () => {
  let finish;
  reorderBubbles.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  render(<PageReview />);
  await screen.findByRole("button", { name: "Déplacer la bulle 1" });
  act(() => {
    drag.onEnd({ active: { id: 11 }, over: { id: 12 } });
  });
  expect(reorderBubbles).toHaveBeenCalledWith("42", [
    { id: 12, order: 1 },
    { id: 11, order: 2 },
  ]);
  expect(
    screen.getByRole("button", { name: "Valider la page" }),
  ).toBeDisabled();
  expect(screen.getAllByRole("listitem")[0]).toHaveTextContent("Deuxième");
  finish({});
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Valider la page" }),
    ).toBeEnabled(),
  );
});
it("restores the previous order when saving fails", async () => {
  reorderBubbles.mockRejectedValue(new Error("network"));
  render(<PageReview />);
  await screen.findByRole("button", { name: "Déplacer la bulle 1" });
  act(() => {
    drag.onEnd({ active: { id: 11 }, over: { id: 12 } });
  });
  await waitFor(() =>
    expect(screen.getAllByRole("listitem")[0]).toHaveTextContent("Première"),
  );
});
it("offers a retry after a load failure", async () => {
  getPageById.mockRejectedValueOnce(new Error("network"));
  render(<PageReview />);
  fireEvent.click(await screen.findByRole("button", { name: "Réessayer" }));
  expect(await screen.findByText("Première")).toBeInTheDocument();
});
it("supports textless pages", async () => {
  getBubblesForPage.mockResolvedValue({ data: [] });
  render(<PageReview />);
  expect(
    await screen.findByText("Aucune bulle sur cette page."),
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Valider la page" })).toBeEnabled();
});

it("does not save cancelled or stationary drops", async () => {
  render(<PageReview />);
  await screen.findByRole("button", { name: "Déplacer la bulle 1" });
  await act(async () => {
    await drag.onEnd({ active: { id: 11 }, over: null });
  });
  await act(async () => {
    await drag.onEnd({ active: { id: 11 }, over: { id: 11 } });
  });
  expect(reorderBubbles).not.toHaveBeenCalled();
});
it("locks dragging after a page has left review", async () => {
  getPageById.mockResolvedValue({
    data: {
      id: 42,
      numero_page: 7,
      statut: "completed",
      url_image: "/page.png",
    },
  });
  render(<PageReview />);
  expect(
    await screen.findByRole("button", { name: "Déplacer la bulle 1" }),
  ).toBeDisabled();
  await act(async () => {
    await drag.onEnd({ active: { id: 11 }, over: { id: 12 } });
  });
  expect(reorderBubbles).not.toHaveBeenCalled();
});
