<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Architecture
- Screening state lives in `src/lib/screening.tsx` (React context) and persists to localStorage; resume PDFs persist in IndexedDB (`src/lib/resume-store.ts`). Why: no backend is connected; data stays in the browser.
- Resume matching is client-side in `src/lib/scoring.ts` (pdfjs text extraction + skill/experience/education weighting). Why: works offline with no AI credentials.
