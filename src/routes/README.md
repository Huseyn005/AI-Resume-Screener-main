# Routes

This Vite SPA uses TanStack Router with file-based route modules. Each page in
this directory is registered in `src/routeTree.gen.ts`; the root route wraps
the pages in `src/routes/__root.tsx`.

## Conventions

| File | URL |
| --- | --- |
| `index.tsx` | `/` |
| `about.tsx` | `/about` |
| `users/index.tsx` | `/users` |
| `users/$id.tsx` | `/users/:id` (dynamic) |
| `posts/{-$category}.tsx` | `/posts/:category?` (optional segment) |
| `files/$.tsx` | `/files/*` (splat) |
| `_layout.tsx` | layout route (renders children via `<Outlet />`) |
| `__root.tsx` | app shell — renders the active route with `<Outlet />` |

When adding or renaming a route, update `src/routeTree.gen.ts` to register it.
