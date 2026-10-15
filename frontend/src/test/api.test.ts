import { afterEach, describe, expect, it, vi } from "vitest";
import { analyzeResumes } from "@/features/screening/api";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("analyzeResumes", () => {
  it.each([
    ["https://screening.ngrok-free.app", true],
    ["https://screening.ngrok-free.dev", true],
    ["https://screening.ngrok.app", true],
    ["https://api.example.com", false],
    ["https://ngrok-free.app.attacker.example", false],
  ] as const)(
    "adds the ngrok warning header for %s only when applicable",
    async (baseUrl, expected) => {
      vi.stubEnv("VITE_API_URL", baseUrl);
      const fetchMock = vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response(JSON.stringify({ results: [] }), { status: 200 }));
      vi.stubGlobal("fetch", fetchMock);

      const resume = new File(["resume"], "candidate.pdf", { type: "application/pdf" });
      await analyzeResumes("Test job description", [resume]);

      const request = fetchMock.mock.calls[0];
      expect(request?.[0]).toBe(`${baseUrl}/analyze`);
      const options = request?.[1];
      const headers = new Headers(options?.headers);
      expect(headers.has("ngrok-skip-browser-warning")).toBe(expected);
      if (expected) {
        expect(headers.get("ngrok-skip-browser-warning")).toBe("true");
      }

      const body = options?.body;
      if (!(body instanceof FormData)) {
        throw new Error("The analyze request body must remain FormData.");
      }
      expect(body.get("job_description")).toBe("Test job description");
      expect(body.getAll("resumes")).toHaveLength(1);
    },
  );

  it("shows the configured API base URL when the backend cannot be reached", async () => {
    const baseUrl = "https://screening.example.test";
    vi.stubEnv("VITE_API_URL", `${baseUrl}/`);
    const fetchMock = vi.fn<typeof fetch>().mockRejectedValue(new TypeError("Failed to fetch"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(analyzeResumes("Test job description", [])).rejects.toThrow(
      `Backend not reachable at ${baseUrl}`,
    );
  });
});
