# Frontend

The React/Vite app is currently rooted at the repository level, with application
code in `src/`. Routes live in `src/routes/`, the shared UI primitives in
`src/components/ui/`, and screening state and API access in `src/lib/`.

Set `VITE_USE_MOCK=true` to use the optional browser-side demo scorer. Normal
screening uses the FastAPI endpoint configured by `VITE_API_URL`.

## Removed optional shadcn primitives

These unreachable primitives were removed because each was the only app code
using the listed third-party package(s). To restore one, run the matching
shadcn command from the frontend package directory and add any required
dependency with npm:

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

Other shared and Radix-only primitives remain available, even where the current
screens do not import them directly.
