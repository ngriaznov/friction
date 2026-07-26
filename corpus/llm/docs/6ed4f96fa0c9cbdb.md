# Programming a DCC decoder with Railtrace

This tutorial walks through the first three things most people want to do with a new command station: hooking up a programming track, reading the CV values already stored in a decoder, and giving that decoder a new address. It assumes you have a command station wired to your layout and a locomotive with a decoder already installed.

## What you need

- A command station with a separate programming track output (usually labelled PROG or PGM).
- A short isolated section of track — 3 feet is plenty — not connected to your main bus.
- Railtrace installed on a laptop, and the USB cable that came with your command station.
- One locomotive you don't mind experimenting on.

## Step 1: Wire the programming track

The programming track must be electrically isolated from the main line. If it branches off your layout, cut both rails with insulated joiners at the entry point, and don't let a locomotive bridge the gap while you're programming — a loco with one truck on each side will short the two outputs together.

Run a pair of feeders from the PROG terminals on the command station to the underside of the isolated section. Solder them; alligator clips work but they cause intermittent reads that are maddening to diagnose. Keep the run under about six feet.

Programming track output is deliberately current-limited, typically to around 250 mA. That is enough to talk to a decoder but not enough to run the motor, so nothing will move while you're on this track. That's normal.

## Step 2: Connect Railtrace

Plug the command station into the laptop and launch Railtrace. On first run it scans for serial ports; pick the one matching your station and click **Connect**. The status bar at the bottom should read `Connected — service mode ready`.

If it says `No response`, check that you selected the right port and that the command station is powered up before the USB cable was plugged in. Some stations enumerate their serial port only at boot.

## Step 3: Read the decoder's current CVs

Put the locomotive on the programming track. In Railtrace, open **Decoder → Read Sheet**.

Railtrace first reads CV8 (manufacturer ID) and CV7 (version). If it recognises the combination it loads a decoder-specific sheet with named fields; if not, you get a generic numbered CV list, which works fine — the numbers mean the same thing either way.

Click **Read All**. Expect this to take a minute or two. Service-mode reads work by asking the decoder a question and watching for a current pulse in reply, one bit at a time, so it is not fast.

The values worth looking at:

- **CV1** — short address, 1–127.
- **CV17/CV18** — long address, stored across two CVs.
- **CV29** — configuration byte, including which address the decoder actually uses.
- **CV2, CV5, CV6** — start, top, and mid voltage.

If reads come back blank or inconsistent, clean the wheels and railheads. Poor pickup is the cause of the great majority of failed reads.

## Step 4: Write a new address

Locomotives numbered above 127 need a long address. In the sheet, type the number into the **Address** field and click **Write**.

Railtrace handles the arithmetic: it splits the address into CV17 and CV18 and sets bit 5 of CV29 so the decoder uses the long address instead of CV1. Doing this by hand is a common source of "the loco won't respond" problems, which is the main reason to let the tool do it.

After writing, click **Read All** once more and confirm the address field shows what you entered. Then move the locomotive to the main line and call it up on the throttle by its new number.
