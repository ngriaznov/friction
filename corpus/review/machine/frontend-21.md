Reviewed the new Storybook stories for the `Button` component's variants (primary, secondary, destructive, ghost, and the loading/disabled states). Quick, clean addition — approving as-is.

The stories use CSF3 format consistently with the rest of the stories directory, args are used properly for the interactive controls panel rather than hardcoding props per story, and I like that you added a `Playground` story with all props exposed as controls in addition to the fixed-variant stories, since that gives consumers of the design system a way to explore combinations that aren't covered by the named stories.

Two small things. The `Loading` story shows the spinner variant but doesn't set `disabled` alongside it, and per the component's actual implementation (checked `Button.tsx`), a loading button is still clickable unless `disabled` is also passed — so the story as written doesn't accurately represent how this state should actually be used in the app, which is presumably always with both props together. Worth adding `disabled: true` to that story's args so it matches real usage, or if there's a legitimate case for loading-without-disabled, a comment noting why would help whoever reads this next.

Also, no story exercises the `onClick` handler being called via an interaction test (`play` function). Not a blocker, since visual stories alone are useful on their own, but given Storybook's interaction testing addon is already installed in this project per `package.json`, a quick `play` function on at least one story confirming the click handler fires and the button isn't clickable while `disabled` would give you a bit of regression coverage for free.

Neither of these needs to hold up merging — good addition to the design system docs either way.
