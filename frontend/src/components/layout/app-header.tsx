import { Link } from "@tanstack/react-router";
import { LogOut, Menu, ScanLine, Settings, Star, UserRound } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useScreening } from "@/features/screening/screening";

const NAV = [
  { to: "/", label: "Screen" },
  { to: "/shortlist", label: "Shortlist" },
  { to: "/jobs", label: "Jobs" },
  { to: "/candidates", label: "Candidates" },
  { to: "/settings", label: "Settings" },
] as const;

const linkCls =
  "rounded-full px-4 py-2 text-sm font-semibold text-muted-foreground transition-colors hover:bg-secondary hover:text-secondary-foreground";
const active = { className: "bg-secondary !text-secondary-foreground" };

export function AppHeader() {
  const { shortlist, profile, signOut } = useScreening();
  const [menu, setMenu] = useState(false);

  const links = (onClick?: () => void, block = false) =>
    NAV.map((n) => (
      <Link
        key={n.to}
        to={n.to}
        onClick={onClick}
        className={block ? `${linkCls} block` : linkCls}
        activeOptions={{ exact: n.to === "/" }}
        activeProps={active}
      >
        <span className="flex items-center gap-1.5">
          {n.to === "/shortlist" && <Star className="size-4" />}
          {n.label}
          {n.to === "/shortlist" && shortlist.length > 0 && (
            <span className="rounded-full bg-accent px-2 text-xs text-accent-foreground">
              {shortlist.length}
            </span>
          )}
        </span>
      </Link>
    ));

  return (
    <header className="sticky top-0 z-30 border-b bg-background/85 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-4">
        <Link to="/" className="flex items-center gap-2 font-display text-xl font-bold">
          <span className="grid size-9 place-items-center rounded-xl bg-hero text-primary-foreground">
            <ScanLine className="size-5" />
          </span>
          shortlist<span className="text-accent">.</span>
        </Link>
        {import.meta.env["VITE_USE_MOCK"] === "true" && (
          <span className="rounded-full bg-warning-soft px-2.5 py-1 text-xs font-semibold text-foreground">
            Demo mode
          </span>
        )}
        <nav className="hidden items-center gap-1 md:flex">{links()}</nav>
        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="secondary"
                size="icon"
                className="rounded-full"
                aria-label="Account menu"
              >
                {profile.name ? (
                  <span className="text-xs font-bold">
                    {profile.name.slice(0, 2).toUpperCase()}
                  </span>
                ) : (
                  <UserRound />
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48 rounded-2xl">
              <div className="px-2 py-1.5 text-sm">
                <p className="font-semibold">{profile.name || "Guest"}</p>
                {profile.email && (
                  <p className="truncate text-xs text-muted-foreground">{profile.email}</p>
                )}
              </div>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link to="/settings">
                  <UserRound className="mr-2 size-4" />
                  Profile
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link to="/settings">
                  <Settings className="mr-2 size-4" />
                  Settings
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={signOut}>
                <LogOut className="mr-2 size-4" />
                Sign out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Button
            variant="ghost"
            size="icon"
            className="rounded-full md:hidden"
            aria-label="Open menu"
            onClick={() => setMenu(true)}
          >
            <Menu />
          </Button>
        </div>
      </div>
      <Sheet open={menu} onOpenChange={setMenu}>
        <SheetContent side="right" className="w-72">
          <SheetTitle className="font-display">Menu</SheetTitle>
          <SheetDescription className="sr-only">Navigate between pages</SheetDescription>
          <nav className="mt-6 space-y-1">{links(() => setMenu(false), true)}</nav>
        </SheetContent>
      </Sheet>
    </header>
  );
}
