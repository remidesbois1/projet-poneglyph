"use client";
import React, { Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import * as Tabs from "@radix-ui/react-tabs";
import { MessageSquareText, Layers, Send } from "lucide-react";
import { useManga } from "@/context/MangaContext";
import styles from "@/components/moderation/Review.module.css";
const BubbleReviewList = React.lazy(
  () => import("@/components/BubbleReviewList"),
);
const PageReviewList = React.lazy(() => import("@/components/PageReviewList"));
const SubmissionsList = React.lazy(
  () => import("@/components/moderation/SubmissionsList"),
);
function ModerationContent() {
  const { currentManga, mangaSlug } = useManga();
  const searchParams = useSearchParams();
  const router = useRouter();
  const view = searchParams.get("view");
  const activeTab = ["pages", "submissions"].includes(view) ? view : "bubbles";
  return (
    <div className={styles.overview}>
      <title>Modération</title>
      <header className={styles.heading}>
        <div>
          <p className={styles.eyebrow}>{currentManga?.titre}</p>
          <h1>Modération</h1>
        </div>
      </header>
      <Tabs.Root
        value={activeTab}
        onValueChange={(value) =>
          router.replace(
            "/" +
              mangaSlug +
              "/moderation" +
              (value === "bubbles" ? "" : "?view=" + value),
            { scroll: false },
          )
        }
        className={styles.tabs}
      >
        <Tabs.List
          className={styles.tabBar}
          aria-label="Modération et suivi des contributions"
        >
          <Tabs.Trigger value="bubbles" className={styles.tab}>
            <MessageSquareText />
            Bulles
          </Tabs.Trigger>
          <Tabs.Trigger value="pages" className={styles.tab}>
            <Layers />
            Pages complètes
          </Tabs.Trigger>
          <Tabs.Trigger value="submissions" className={styles.tab}>
            <Send />
            Mes Soumissions
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="bubbles" className={styles.tabPanel}>
          <Suspense
            fallback={
              <p className={styles.empty} role="status">
                Chargement des bulles…
              </p>
            }
          >
            <BubbleReviewList />
          </Suspense>
        </Tabs.Content>
        <Tabs.Content value="submissions" className={styles.tabPanel}>
          <Suspense
            fallback={
              <p className={styles.empty} role="status">
                Chargement des soumissions…
              </p>
            }
          >
            <SubmissionsList />
          </Suspense>
        </Tabs.Content>
        <Tabs.Content value="pages" className={styles.tabPanel}>
          <Suspense
            fallback={
              <p className={styles.empty} role="status">
                Chargement des pages…
              </p>
            }
          >
            <PageReviewList />
          </Suspense>
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
}

export default function ModerationPage() {
  return (
    <Suspense fallback={<p role="status">Chargement…</p>}>
      <ModerationContent />
    </Suspense>
  );
}
