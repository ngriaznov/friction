**Subject:** Request: accessible date-range picker for the shared component library

Hi all,

I'd like to formally request a date-range picker component for the shared library. Three of our internal tools have independently built one in the last eighteen months, all three have accessibility problems, and we're about to need a fourth. I'd rather stop that before it happens.

Below is the case, the requirements as I understand them, and a date I'm hoping we can work toward. Happy to be wrong about any of the specifics — this is a starting point, not a spec handed down.

**Where it's needed**

*Billing Console.* Invoice search filters by issue date range. Currently two separate `<input type="date">` fields with a bit of validation glue. It works, but there's no relationship between the two inputs — you can set an end date before the start date and only find out when the search returns nothing. No preset ranges, which is the most common complaint from the finance team since they almost always want "last month" or "this quarter."

*Fleet Dashboard.* Telemetry charts scoped to a window. This one uses a third-party picker that we pulled in during a crunch. It's the worst of the three from an accessibility standpoint: the calendar grid is a set of divs with click handlers, no roles, and it traps focus when opened via keyboard. Our last audit flagged it and we've been carrying the finding for two quarters.

*Support Triage.* Ticket volume reports. Home-grown, keyboard-navigable, but the announcement behaviour is wrong — screen reader users hear the day number and nothing else, so there's no way to know which month you're in while arrowing around.

The fourth is the workforce scheduling tool the platform team is starting next quarter. They've already asked me what we use.

**What we need it to do**

Functionally: select a start and end date, with the constraint enforced in the component rather than by each consumer. Optional preset ranges supplied by the consumer (last 7 days, last 30, this month, last month, this quarter, custom). Optional min/max bounds. Optional single-month vs. two-month display, since Billing has room for two and Support Triage doesn't.

Keyboard interaction is the part I care most about, so I'll be specific about what we need:

- Arrow keys move by day, PageUp/PageDown by month, Shift+PageUp/PageDown by year, Home/End to start/end of week.
- Enter or Space commits the focused date. First commit sets the start, second sets the end. Committing a date earlier than the current start should reset the selection to a new start rather than erroring.
- Escape closes the picker and returns focus to the trigger, discarding an in-progress range.
- Tab moves through the picker's controls (month nav, preset list, grid) and out — no focus trap, since this isn't a modal.
- The date grid should be a real `role="grid"` with `aria-selected` on cells in range, and the focused cell should carry an accessible name including the full date ("Tuesday, March 14, 2026"), not just the day number.
- Range boundaries and in-range days need to be distinguishable without colour alone.
- Typing into the text inputs should stay supported. Several of our power users never open the calendar at all, and I'd hate to lose that.

Locale and timezone: all three tools currently assume the user's local timezone and en-US formatting. I'd suggest the component take a locale prop and stay timezone-agnostic (dates in, dates out, no instants), but that's your call and I'd defer to whatever the library does elsewhere.

**Timing**

Fleet Dashboard's accessibility finding comes up for review at the end of Q2, and I'd like to close it with the shared component rather than another patch. That means having something usable — even an alpha we can integrate behind a flag — by **May 15**, with a stable release whenever it's ready after that.

If that's not realistic, tell me and I'll plan around it. If it helps, I can put a couple of weeks into building it with your guidance rather than waiting on your queue — I've done the keyboard-interaction work twice now and would rather it land somewhere permanent.

Let me know what you need from me. I can pull screenshots of all three current implementations, or walk through them live if that's more useful.

Thanks,
Priya
