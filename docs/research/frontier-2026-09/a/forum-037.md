**Q: How do I get started with room correction for a stereo pair? Where should I sit, and how do I measure?**

---

**A:**

Room correction has three layers, and you should work through them in order: **position first, measure second, EQ last.** Most people jump to EQ and then wonder why a 15 dB hole at 60 Hz won't fill in.

### 1. Start with the listening position: the 38% rule

The simplest improvement you can make is to move your chair. A common starting point is to put your ears about **38% of the room's length** from the front or rear wall.

The reason is room modes. The axial modes along the length of a room set up standing waves at

```
f = n × c / (2L)
```

where *c* ≈ 343 m/s and *L* is the room length. In a 5.5 m room the first length mode is about 31 Hz, the second about 62 Hz, and so on. Each mode has pressure peaks at the walls and nulls at predictable points in between: the first-order mode has its null at 50% of the length, the second at 25% and 75%, and so on. Sitting at the exact centre of the room puts you in the null of every odd-order length mode, which is why the "sweet spot in the middle of the room" often sounds thin.

At 38% you avoid the worst peaks and nulls of the first few modes. Treat it as a starting point, not a law. Width and height modes still matter, and so do real walls that aren't perfectly rigid. Put your chair at 38%, then measure, then try positions 10–20 cm either side.

### 2. Measuring with REW

You'll need:

- **A calibrated measurement mic.** A USB mic like the miniDSP UMIK-1 is the easy option because it comes with a serial-specific calibration file.
- **REW (Room EQ Wizard).** It's free.
- A way to get audio from the computer to your amp. HDMI or a USB DAC both work. Set up REW's timing reference or loopback if your interface supports it.

Procedure:

1. Load the mic's calibration file under *Preferences → Cal files*.
2. Point the mic straight up (use the 90° cal file if one is provided) at ear height, on a stand, exactly where your head goes.
3. Use REW's level-check to set a sweep level of roughly **75–80 dB SPL**. That's loud enough to rise well above the noise floor, but not so loud that the speakers or room rattles distort.
4. Run a **log swept sine** from 20 Hz to 20 kHz, at 256k or 512k length. Longer sweeps give better signal-to-noise in the bass.
5. Measure the **left and right speakers separately**, then both together.
6. Take several measurements, moving the mic a few centimetres each time, and average them (*All SPL → Average the responses*). One single-point measurement will send you chasing features that only exist in a 2 cm spot.

When you look at the bass, use 1/12 or 1/24 octave smoothing. 1/3 octave smoothing will hide exactly the problems you're trying to find.

### 3. What a room-mode null looks like

Plot 20–300 Hz. A **modal peak** is a broad-ish bump, often 6–12 dB above the trend line. A **null** looks different:

- It's a **deep, narrow notch**, often 10–25 dB down, with steep walls either side. On a 1/24-octave graph it looks like a V or a spike pointing downward.
- It **moves or changes depth a lot** when you shift the mic 20–30 cm, while the overall trend stays put.
- It often lands near a frequency you can predict from the room dimensions (compare against REW's *Room Sim* or a mode calculator).
- In the **waterfall** or spectrogram view, a peak shows a long ringing decay ridge. A null shows as a gap with little energy at all.

### Don't boost the nulls

A null is a cancellation: the direct sound and a reflection arriving out of phase. Adding 10 dB of EQ adds 10 dB to *both*, so they still cancel. All you do is burn amplifier headroom and push the woofer's excursion. Fix nulls by **moving the seat or the speakers** (or adding a sub in another position). Keep EQ for **cutting peaks**, which it does well.

A sensible workflow: move the seat to about 38%, measure, adjust the speaker positions, measure again, and when you have the smoothest average, use REW's EQ module to cut the remaining peaks below roughly 300 Hz only. Set the filters to cut only and leave the treble alone.
