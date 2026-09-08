"use client";
import React, { Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import * as Tabs from "@radix-ui/react-tabs";
import { MessageSquareText, Layers } from "lucide-react";
import { useManga } from "@/context/MangaContext";
import styles from "@/components/moderation/Review.module.css";
const BubbleReviewList = React.lazy(
  () => import("@/components/BubbleReviewList"),
);
const PageReviewList = React.lazy(() => import("@/components/PageReviewList"));
function ModerationContent() {
  const { currentManga, mangaSlug } = useManga();
  const searchParams = useSearchParams();
  const router = useRouter();
  const activeTab = searchParams.get("view") === "pages" ? "pages" : "bubbles";
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
              (value === "pages" ? "?view=pages" : ""),
            { scroll: false },
          )
        }
        className={styles.tabs}
      >
        <Tabs.List
          className={styles.tabBar}
          aria-label="Contributions à vérifier"
        >
          <Tabs.Trigger value="bubbles" className={styles.tab}>
            <MessageSquareText />
            Bulles
          </Tabs.Trigger>
          <Tabs.Trigger value="pages" className={styles.tab}>
            <Layers />
            Pages complètes
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
