It starts with *spaces* in the panel
----

A service panel has a basic unit I call a *space*.  That's apparent when you look at the "knockouts" on the cover.

Electrical power in North America delivers split-phase power. Meaning there are two "legs" or "poles" - L1 and L2 - both 120V from neutral, and opposite-phased so they add up to 240V.

[![Pic: X-ray of service panel showing bus-bars alternating rows][1]][1]

Here's an X-ray of a service panel cover.  See the trick?  The buses are arranged in alternating rows. Every other row is a different pole *on both sides*. Any single-space breaker gets only one pole and 120V (its circuit also attaches to the neutral bar, not shown) -- but a 2-space breaker has access to both poles and 240V.

A single-space breaker cannot get 240V.  That's very important when we get to duplex breakers.  *Now let's fit some breakers.*

[![Pic: striped for legs with breakers in][2]][2]

On the upper right you see a **single breaker**. This occupies one space and serves one circuit.

On the upper left, you see a **2-pole breaker**: this occupies two spaces, and can access both poles L1 and L2.  Both sides will trip together if either side overloads.  This is used for 240V loads, or for multi-wire branch circuits (MWBC).

[![enter image description here][3]][3]

Backside of a 2-pole or quad breaker.  Note the 2 clips grabbing 2 different poles.

Duplex and friend
----

On the lower right is the star of the show: the **duplex/tandem breaker**. This occupies a single space and can **only access a single pole**.  The sides will not trip together.   It cannot power 240V loads, it *must not* power MWBCs, and it is generally used for two unrelated circuits.  The entire point of the duplex is to save space in the box.  We often call that a "double-stuff" for obvious reasons, and to emphasize how it's different from a 2-pole.

[![enter image description here][4]][4]

*Back of a duplex breaker. Only one clip to grab one pole.*

Lower left is the **quad breaker**, a 2-pole duplex breaker. Again the purpose is to save space.  It's just what it looks like, 2 duplexes, except it can access both poles L1 and L2, and it has handle-ties so the inner pair will trip together on overload.  That makes it suitable for a 240V load or MWBC.  On this model, the outside pair is also tied together with common trip and can serve 240V loads or MWBC.  Or, you can get these breakers with outer breakers independent.

General Electric's Q-line panels take a different approach to the problem, and you will never find a quadplex in Q-line.

No substitute for spaces
----

Most panels only allow these double-stuff breakers in certain positions.  Some only forbid it in their labeling, while others actually key the buses so duplex breakers won't fit.  Don't force them.

Nowadays in new or remodel work, most circuits must be GFCI and/or AFCI.  Here's the trick: for most panels, GFCI/AFCI breakers *are not available* in duplex/tandem/double-stuff. This is why you can no longer add circuits by switching to these, and why being low on breaker spaces is kind of a big deal.  Which is why many of us recommend quite large panels.

The 12-space panel featured here for illustration is terribly small, more befitting a shed than a house.  They are made as large as 60 or even 84 space.  Many will say "40 circuits/30 spaces" meaning they can accommodate double-stuff's in 10 spaces, but only have 30 total. Watch out for that.

  [1]: https://i.sstatic.net/Nrrv9.png
  [2]: https://i.sstatic.net/DDz1l.png
  [3]: https://i.sstatic.net/gf7wY.jpg
  [4]: https://i.sstatic.net/mTnef.jpg
