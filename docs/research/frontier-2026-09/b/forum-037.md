Room correction is really three separate jobs: put the speakers and chair somewhere reasonable, measure what the room is doing to the sound, and only then decide whether EQ or treatment is worth it. Skipping the first two and jumping straight to a correction filter is how people end up "fixing" problems that a chair moved 40 cm would have solved.

## 1. Start with placement: the 38% rule

In a rectangular room, the worst bass problems come from standing waves along the room's length. The listening position matters as much as the speaker position here. A widely used rule of thumb (usually credited to Wes Lachot) is to put your ears about **38% of the room length** from the front wall, on the centreline. So in a 5 m long room, sit about 1.9 m from the front wall.

Why 38%? Sitting at 50% puts you at the pressure null of the second-order length mode and the peak of the first, which is the worst of both worlds. Sitting at 25% lands you on other nodes. 38% is simply a spot that is not on an integer fraction of the length for the first several modes, so no single mode dominates. It is not magic and it is not exact; treat it as a starting point and expect to move ±20 cm after measuring.

Speakers go roughly equidistant from the side walls, with the tweeters and the listening position forming an equilateral triangle, and pulled away from the front wall if at all possible.

## 2. Measuring with REW

Room EQ Wizard is free, and a calibrated USB measurement mic (UMIK-1 or similar) is about the only thing you need to buy. Rough procedure:

1. Load the mic calibration file in **Preferences > Mic/Meter**, and set the sample rate to match your interface.
2. Put the mic at ear height at the listening position, pointed at the ceiling (use the 90° cal file) or at the speakers (0° file). Pick one and stay consistent.
3. **Measure**, select the swept sine ("Sweep"), 20 Hz to 20 kHz, length 256k or 512k. Measure each speaker individually, then both together.
4. Check the level: REW will complain if the sweep is too quiet or clipping. Aim for the signal to be roughly 75–80 dB SPL at the mic.
5. Take a couple of sweeps per position and compare them. If they don't overlay closely, something is moving (you, the mic stand, a fan).

Once you have a measurement, apply **1/6 or 1/12 octave smoothing** for the bass region. Unsmoothed curves look terrifying and are not what you hear.

## 3. What a room-mode null actually looks like

This is the part that catches people. On the SPL graph, a **peak** from a room mode is a hump of maybe 6–10 dB, fairly broad. A **null** is much more dramatic: a narrow, deep notch, often 15–25 dB down, at a specific frequency. In a 5 m room the first length mode is around 34 Hz and its null will typically show up somewhere between 60 and 90 Hz depending on where you sit.

Three things identify it as a modal null rather than a measurement error:

- It is **narrow and sharp**, not a gentle dip.
- It **moves in frequency or depth when you move the mic** 30–50 cm forward or back. Speaker-boundary interference does this too, but a room-mode null is tied to the room dimensions, so it stays at the same frequency and only changes depth.
- On the **waterfall or spectrogram** plot, the frequencies either side of the null ring for a long time (long decay) while the null itself decays quickly, because there is simply no energy there at that position.

The important consequence: **you cannot EQ a null.** Boosting 20 dB at 75 Hz just burns amplifier power and driver excursion to fill a hole that the room cancels out again. The fixes for a null are moving the chair, moving the speakers, adding a subwoofer at a different position, or bass trapping. EQ is for pulling down the peaks afterwards.

So: place, measure, move, re-measure. When the remaining problems are peaks rather than holes, that's the point to reach for correction.
