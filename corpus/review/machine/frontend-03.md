A few things stand out in this dashboard grid layout. The core `display: grid` setup with named template areas for `sidebar`, `header`, `main`, and `footer` is clean and genuinely easier to read than the equivalent nested flexbox would be, so I'd keep that part as-is.

The problem is the media query breakpoints. You're collapsing to a single-column layout at `max-width: 768px` by redefining `grid-template-areas`, which is correct, but you never reset `grid-template-columns`, so the three-column `250px 1fr 1fr` definition from the desktop layout is still active underneath the mobile stack. On a narrow viewport this produces a layout where the areas are logically stacked but the columns they sit in are still sized for desktop, causing visible horizontal overflow and a scrollbar that shouldn't be there. Add `grid-template-columns: 1fr;` inside the same media query block.

Second, you're mixing units inconsistently — `gap` is in `rem`, padding on `.dashboard-main` is in `px`, and the sidebar width is a hardcoded `250px`. Not a bug, but pick one system and stick to it; mixing `px` and `rem` in the same component makes it harder to reason about how the layout responds to a user's browser font-size setting, and `250px` for the sidebar should probably be a CSS custom property (`--sidebar-width`) since it's referenced in both the grid template and, apparently, a JS resize handler elsewhere.

Also worth checking: `.dashboard-main` has `overflow-y: auto` but no `min-height: 0`. Grid children default to `min-height: auto`, which means the content can force the row to grow past its track size rather than actually scrolling internally — a very common gotcha with grid/flex children that need internal scroll regions. Add `min-height: 0` (or `min-width: 0` for row-direction cases) to `.dashboard-main`.

Small nit: `z-index: 999` on the header. Pick a real scale (10, 20, 30…) rather than reaching for four nines; it's a sign the number was chosen to "just win" rather than deliberately placed relative to other layered elements, and it'll cause pain the day something needs to sit above it.

Otherwise solid — approve once the column reset and `min-height: 0` are in.
