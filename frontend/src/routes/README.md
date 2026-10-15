# Application routes

The TanStack Router route modules are:

| File | URL | Purpose |
| --- | --- | --- |
| `index.tsx` | `/` | Start a resume screening |
| `jobs.tsx` | `/jobs` | Manage saved jobs |
| `candidates.tsx` | `/candidates` | Browse candidates across jobs |
| `shortlist.tsx` | `/shortlist` | Review and filter shortlisted candidates |
| `settings.tsx` | `/settings` | Profile, scoring weights, and local data |
| `__root.tsx` | Root layout | Shared app shell and route outlet |

`frontend/src/routeTree.gen.ts` is maintained by hand in this project; there is no
TanStack Router Vite plugin generating it. When adding, removing, or renaming a
route, update its imports, route registrations, and route type declarations
alongside the route module. Keep the route matching test in `frontend/src/test/` passing.
