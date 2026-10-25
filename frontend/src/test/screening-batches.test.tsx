import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ScreeningProvider, useScreening } from "@/features/screening/screening";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

function wrapper({ children }: { children: ReactNode }) {
  return <ScreeningProvider>{children}</ScreeningProvider>;
}

function analysis(filename: string) {
  return {
    filename,
    candidate_name: filename,
    match_percentage: 80,
    top_skills: [],
    missing_skills: [],
    matched: [],
    skills: [],
    score_breakdown: { semantic: 80, skills: 80, section_fit: 80 },
    reason: "Analysis completed.",
    flags: [],
    status: "ok",
    role: "",
    experience: "",
    strengths: [],
    gaps: [],
    summary: "Analysis completed.",
  };
}

function uploadNames(init: RequestInit | undefined): string[] {
  const body = init?.body;
  if (!(body instanceof FormData)) throw new Error("Expected FormData.");
  return body.getAll("resumes").map((file) => (file as File).name);
}

describe("screening batches", () => {
  it("uploads at most five files sequentially and reports progress per batch", async () => {
    vi.stubEnv("VITE_USE_MOCK", "false");
    vi.stubEnv("VITE_API_URL", "https://screening.example.test");
    const fetchMock = vi.fn<typeof fetch>(async (_input, init) => {
      const names = uploadNames(init);
      return new Response(
        JSON.stringify({ results: names.map((name) => analysis(name)).reverse() }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useScreening(), { wrapper });
    const files = Array.from(
      { length: 12 },
      (_, index) =>
        ({
          id: `resume-${index}`,
          name: `resume-${index}.pdf`,
          size: 10,
          file: new File(["pdf"], `resume-${index}.pdf`, { type: "application/pdf" }),
        }) satisfies (typeof result.current.files)[number],
    );
    act(() => {
      result.current.setJD("A sufficiently long job description for testing.");
      result.current.setFiles(files);
    });
    const progress: number[] = [];

    await act(async () => {
      await result.current.runScreening((done) => progress.push(done));
    });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls.map(([, init]) => uploadNames(init))).toEqual([
      files.slice(0, 5).map((file) => file.name),
      files.slice(5, 10).map((file) => file.name),
      files.slice(10).map((file) => file.name),
    ]);
    expect(progress).toEqual([5, 10, 12]);
    expect(result.current.candidates).toHaveLength(12);
    expect(
      result.current.candidates.find((candidate) => candidate.resumeId === "resume-7")?.name,
    ).toBe("resume-7.pdf");
  });

  it("keeps successful batches and creates error candidates for failed batches", async () => {
    vi.stubEnv("VITE_USE_MOCK", "false");
    vi.stubEnv("VITE_API_URL", "https://screening.example.test");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockImplementationOnce(async (_input, init) => {
        return new Response(
          JSON.stringify({
            results: uploadNames(init).map((name) => analysis(name)),
          }),
        );
      })
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "The backend is unavailable." }), {
          status: 400,
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useScreening(), { wrapper });
    const files = Array.from({ length: 6 }, (_, index) => ({
      id: `resume-${index}`,
      name: `resume-${index}.pdf`,
      size: 10,
      file: new File(["pdf"], `resume-${index}.pdf`, { type: "application/pdf" }),
    }));
    act(() => {
      result.current.setJD("A sufficiently long job description for testing.");
      result.current.setFiles(files);
    });
    const progress: number[] = [];
    let candidates: Awaited<ReturnType<typeof result.current.runScreening>> = [];

    await act(async () => {
      candidates = await result.current.runScreening((done) => progress.push(done));
    });

    expect(progress).toEqual([5, 6]);
    expect(candidates.filter((candidate) => candidate.status === "ok")).toHaveLength(5);
    expect(candidates.filter((candidate) => candidate.status === "error")).toHaveLength(1);
    expect(candidates.find((candidate) => candidate.status === "error")?.reason).toContain(
      "backend is unavailable",
    );
  });

  it("throws when all batches fail", async () => {
    vi.stubEnv("VITE_USE_MOCK", "false");
    vi.stubEnv("VITE_API_URL", "https://screening.example.test");
    vi.stubGlobal(
      "fetch",
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: "Service unavailable." }), { status: 400 }),
        ),
    );
    const { result } = renderHook(() => useScreening(), { wrapper });
    act(() => {
      result.current.setJD("A sufficiently long job description for testing.");
      result.current.setFiles([
        {
          id: "resume-1",
          name: "resume-1.pdf",
          size: 10,
          file: new File(["pdf"], "resume-1.pdf", { type: "application/pdf" }),
        },
      ]);
    });

    await expect(act(async () => result.current.runScreening(() => undefined))).rejects.toThrow(
      "All resume batches failed",
    );
  });
});
