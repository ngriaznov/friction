# Flashing and Calibrating Your Line-Follower with Trundlebot Console

This tutorial takes a freshly assembled line-following robot from a blank motor controller to a tuned, saved calibration. You will flash the firmware, tune the PID loop that keeps the robot on the line, and store the result as a profile so the settings survive a power cycle.

Plan for about 45 minutes, including a few test runs on the track.

## What you need

- A club robot with the Trundlebot motor controller board and the five-sensor line array attached.
- A USB-C cable that carries data (some charging cables do not).
- A laptop with Trundlebot Console installed. Download it from the club shared drive if you do not have it yet.
- A section of test track with a black line on a white surface, including at least one curve.
- A fully charged battery. Low voltage makes tuning results unreliable.

## Step 1: Connect and flash the firmware

1. Switch the robot off and connect it to your laptop with the USB cable.
2. Open Trundlebot Console. In the top bar, choose the port that appears when you plug in the robot (on Windows it looks like `COM5`; on macOS and Linux it looks like `/dev/tty.usbmodem...` or `/dev/ttyACM0`).
3. Switch the robot on. The **Board** panel should show the controller's serial number and its current firmware version, or `no firmware` on a new board.
4. Open the **Firmware** tab and click **Load bundle**. Pick the latest `trundlebot-linefollower-x.y.z.tbf` file from the shared drive.
5. Click **Flash**. Keep the cable plugged in and do not switch the robot off. The progress bar takes about 20 seconds, then the board reboots.
6. Confirm that the **Board** panel now reports the new version.

If flashing fails partway, hold the **BOOT** button on the controller while switching it on, then repeat from step 4. This forces the board into its bootloader.

## Step 2: Check the sensors

Before tuning, make sure the line sensors see what you expect.

1. Open the **Sensors** tab. Five bars show the live reading from each sensor, from left to right.
2. Hold the robot over white surface. All bars should be low.
3. Slide the robot sideways so the line passes under the array. Each bar should rise as the line passes beneath its sensor.
4. Click **Calibrate sensors** and sweep the robot across the line a few times while the timer counts down. Trundlebot Console records the minimum and maximum for each sensor and normalizes them so a partly shadowed sensor still reads correctly.

If a bar never moves, check that sensor's cable before continuing.

## Step 3: Tune the PID loop

The controller computes an *error* from the sensor array: zero when the line is centred, negative when it drifts left, positive when it drifts right. The PID loop turns that error into a difference in motor speeds. Three gains shape how it responds:

- **P (proportional)** steers harder the further off the line the robot is.
- **I (integral)** slowly corrects a steady drift, such as one motor being slightly weaker.
- **D (derivative)** resists sudden changes, which damps side-to-side wobble.

Open the **Tuning** tab. Set a **Base speed** of about 40% for the first runs. Then:

1. Set **I** and **D** to `0`. Set **P** to `0.5`.
2. Place the robot on the track and click **Run**. The robot drives for the number of seconds in the **Run length** box, then stops. Watch what happens.
3. If it drifts off the line on curves, increase **P** by `0.2` and run again. If it snakes back and forth, you have gone too far; reduce **P** slightly.
4. Once it follows the line but wobbles, add **D**. Start at `1.0` and increase in steps of `0.5` until the wobble settles. Too much **D** makes the motion jittery.
5. If the robot consistently rides one side of the line, add a small **I**, around `0.01`. Increase only until the offset disappears; large **I** values cause slow, wide oscillation.
6. Raise **Base speed** in 10% steps and repeat. Faster runs usually need slightly higher **P** and **D**.

The **Trace** graph below the controls plots error over time for the last run. A good tune shows a line that stays near zero with small, quickly decaying bumps at each curve.

## Step 4: Save a calibration profile

Gains and sensor calibration are held in RAM until you save them.

1. In the **Profiles** panel, click **Save as**.
2. Name the profile something you will recognise later, such as `club-track-40pct` or `competition-fast`.
3. Click **Save**. The profile is written to the controller's flash and also copied to your laptop under `~/TrundlebotConsole/profiles/`.
4. Set it as the **Startup profile** so the robot loads it when powered on without a laptop.

You can keep several profiles for different tracks or speeds and switch between them from the same panel. Use **Export** to share a profile file with another club member.

## Troubleshooting

- **Robot spins in place:** one motor is wired backwards. Open **Motors** and tick **Invert** for that side.
- **Robot ignores the line entirely:** re-run sensor calibration; the earlier one may have been done under different lighting.
- **Console cannot find the port:** try a different cable, then a different USB port.
