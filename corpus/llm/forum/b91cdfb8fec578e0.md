**Subject: C64 PSU — recapped it, load tested it, then built a switcher anyway (long)**

Posting this partly as a build log and partly because the "just recap the brick" advice gets repeated a lot and I don't think it's good advice. Bear with me.

**Symptom**

Breadbin C64, assy 250466. Powered up to a black screen. No border, no chirp from the SID, nothing. Checked the 9V AC rail at the board — fine, 9.4V AC unloaded. The 5V rail measured 4.1V at the edge connector and sagged further the moment I touched anything.

The brick is one of the potted Commodore units, the ones filled with epoxy that people describe as "the killer." Mine is a 251053-02.

**Getting into it**

You can't service these gracefully. The epoxy is not a thin coat, it's the entire interior volume. I ended up doing the standard thing: hacksaw down the seam of the case, split the shell off in pieces, then heat gun at about 200°C and a chisel to chip epoxy off the board in flakes. It took two evenings and the smell was awful. Do it outside. Wear a respirator that's actually rated for organic vapour, not a dust mask.

Underneath: a small mains transformer, a bridge rectifier, a 7805-class regulator bolted to the case for heatsinking, and three electrolytics. Two of the three had visibly lifted vents and one had leaked onto the board.

**The recap**

Replaced all three electrolytics with 105°C Panasonic parts, same values, higher voltage rating where the footprint allowed. Cleaned the leaked electrolyte off with isopropyl and a brush, checked the traces underneath with a magnifier — one was thinned but continuous, so I reinforced it with a bit of wire.

Reassembled loosely (couldn't fully re-pot it, obviously) and put it on the bench.

**Load testing — this is the part people skip**

Unloaded, it read 5.02V. Beautiful. If I'd stopped there I'd have posted a triumphant "fixed it!" reply and put it back into service.

I have a cheap constant-current electronic load, and I pulled 1.5A from it, which is roughly what a C64 with a couple of cartridges draws. Output dropped to 4.78V within about ninety seconds and kept drifting down. Scope on the rail showed roughly 180mV of ripple and, more worryingly, the regulator case was at 96°C measured with a thermocouple.

So the caps weren't the whole story. The pass element in these has been cooking inside epoxy for forty years, and the thermal path was always marginal by design. Mine was degraded and there is no way to know how degraded without destructive testing.

**Why I stopped there**

Here's my reasoning, and this is the bit I'd argue about.

The failure mode of a C64 PSU is not "it stops working." That would be fine. The failure mode is the 5V regulator failing *short*, dumping unregulated ~12V into the machine. That kills the PLA, the SID, and frequently the RAM. Those are the exact chips that are hardest and most expensive to replace, and on a 250466 the PLA is not socketed on every revision.

So the question isn't "can I make this brick work today." It's "am I confident this forty-year-old series regulator, whose thermal history I cannot inspect, will never fail short over the next twenty years of use." I'm not. Recapping fixes the ripple. It does nothing about the regulator.

There's also a sentimental argument for keeping the original, and I respect it — but I'd rather keep the original brick on a shelf, intact and unpowered, than saw it apart and pretend the repair restored something.

**The replacement**

Built a modern unit into a project box:

- 5V rail: a Meanwell RS-25-5 module. 5A, way past the ~1.7A the machine ever wants, so it loafs. Adjusted to 5.05V at the connector to account for cable drop.
- 9V AC rail: has to stay AC, the machine uses it for the SID and the TOD clock timing, so a small 9V AC transformer, 1A. No rectification, straight through.
- Crowbar on the 5V: SCR plus a 5.6V zener across the rail. If the 5V ever exceeds about 6.2V it shorts the supply and blows the fuse. This is the whole point of the exercise. A 5A fast-blow in line.
- DIN connector salvaged from the original brick's cable.

Measured under 1.5A load: 5.04V, 22mV ripple, everything under 40°C after an hour.

Machine boots, 38911 BASIC BYTES FREE, ran a memory test for two hours clean.

Happy to post the crowbar schematic if there's interest. It's four components and it's the cheapest insurance in this hobby.
