# Frontend

The React/Vite app lives in `frontend/`:

- `src/features/screening/` contains the API client, screening context, resume
  storage, candidate components, and opt-in mock scorer.
- `src/components/layout/` contains shared app layout components.
- `src/components/ui/` contains shared UI primitives.
- `src/routes/` contains the five pages and root layout.
- `src/test/` contains browser-level tests.

`src/routeTree.gen.ts` is maintained by hand; no TanStack Router Vite plugin is
configured. Keep its route imports and type declarations in sync with route
modules.

Set `VITE_USE_MOCK=true` to use the optional browser-side demo scorer.
Otherwise, screening posts the job description and resumes to the FastAPI URL
configured by `VITE_API_URL`.

## Removed optional shadcn primitives

These unreachable primitives were removed because each was the only app code
using the listed third-party package(s). To restore one, run the matching
shadcn command from `frontend/` and add any required package with npm:

| Removed primitive | Removed package(s) | Regeneration command |
| --- | --- | --- |
| `chart` | `recharts` | `npx shadcn@latest add chart` |
| `carousel` | `embla-carousel-react` | `npx shadcn@latest add carousel` |
| `drawer` | `vaul` | `npx shadcn@latest add drawer` |
| `command` | `cmdk` | `npx shadcn@latest add command` |
| `input-otp` | `input-otp` | `npx shadcn@latest add input-otp` |
| `calendar` | `react-day-picker`, `date-fns` | `npx shadcn@latest add calendar` |
| `resizable` | `react-resizable-panels` | `npx shadcn@latest add resizable` |
| `form` | `react-hook-form`, `@hookform/resolvers`, `zod` | `npx shadcn@latest add form` |

Other shared and Radix-only primitives remain available, even where the
current screens do not import them directly.
