import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useRouter,
  useRouterState,
  type ErrorComponentProps,
} from "@tanstack/react-router";
import { useEffect } from "react";

import { Toaster } from "@/components/ui/sonner";
import { ScreeningProvider } from "@/features/screening/screening";
import { AppHeader } from "@/components/layout/app-header";

const pageMetadata: Record<string, { title: string; description: string }> = {
  "/": {
    title: "Screen resumes — Shortlist",
    description: "Paste a job description, upload resumes, and see who fits best.",
  },
  "/candidates": {
    title: "All candidates — Shortlist",
    description: "Search every candidate across all of your saved jobs.",
  },
  "/jobs": {
    title: "Saved jobs — Shortlist",
    description: "Reopen saved job descriptions and their screened candidates.",
  },
  "/settings": {
    title: "Settings — Shortlist",
    description: "Manage your profile, scoring weights and screening data.",
  },
  "/shortlist": {
    title: "Your shortlist — Shortlist",
    description: "Candidates ranked by match score, with filters and CSV export.",
  },
};
const defaultMetadata = {
  title: "Shortlist — Resume screening",
  description: "Match resumes to your job in three simple steps.",
};

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-7xl font-bold text-foreground">404</h1>
        <h2 className="mt-4 text-xl font-semibold text-foreground">Page not found</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="mt-6">
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Go home
          </Link>
        </div>
      </div>
    </div>
  );
}

function ErrorComponent({ error, reset }: ErrorComponentProps) {
  console.error(error);
  const router = useRouter();

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-xl font-semibold tracking-tight text-foreground">
          This page didn't load
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Something went wrong on our end. You can try refreshing or head back home.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <button
            onClick={() => {
              router.invalidate();
              reset();
            }}
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Try again
          </button>
          <a
            href="/"
            className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
          >
            Go home
          </a>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
  errorComponent: ErrorComponent,
});

function RootComponent() {
  const { queryClient } = Route.useRouteContext();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const metadata = pageMetadata[pathname] ?? defaultMetadata;

  useEffect(() => {
    document.title = metadata.title;
    document
      .querySelector('meta[name="description"]')
      ?.setAttribute("content", metadata.description);
    document.querySelector('meta[property="og:title"]')?.setAttribute("content", metadata.title);
    document
      .querySelector('meta[property="og:description"]')
      ?.setAttribute("content", metadata.description);
  }, [metadata.title, metadata.description]);

  return (
    <QueryClientProvider client={queryClient}>
      <ScreeningProvider>
        <AppHeader />
        <Outlet />
        <Toaster richColors position="top-center" />
      </ScreeningProvider>
    </QueryClientProvider>
  );
}
