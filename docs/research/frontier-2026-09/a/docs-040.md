# Flashing and Calibrating Your Line-Following Robot with Trundlebot Console

In this tutorial, you'll load fresh firmware onto your robot's motor controller, tune it so it follows a line smoothly, and save the settings so you don't have to start over next meeting. Plan on about 45 minutes, and most of that is tuning.

## What you'll need

- A line-following robot with a Trundlebot motor controller board
- A USB cable (micro-USB or USB-C, depending on your board revision)
- A laptop with **Trundlebot Console** installed (Windows, macOS, or Linux)
- A test track: black electrical tape on a light surface, with at least one straight section and one curve
- Fully charged batteries. Low batteries make the motors behave differently and will throw off your tuning.

## Part 1: Flash the firmware

Flashing replaces the program on the motor controller. Do this when you get a new board, when the club releases a firmware update, or when a board is acting strangely.

1. **Prop the robot up** so its wheels are off the table. The motors may twitch during flashing.
2. Connect the controller to your laptop with the USB cable and open Trundlebot Console.
3. Click **Connect** in the top bar. Choose your board's port from the list. On Windows it looks like `COM3`, and on macOS or Linux it looks like `/dev/ttyUSB0` or `/dev/cu.usbserial-…`. If you don't see the board, try a different cable. Some cables only carry power.
4. Open the **Firmware** tab. The console shows the version currently on the board.
5. Click **Check for updates** and select the latest **Line Follower** firmware. If your club keeps its own build, click **Load from file…** and select the `.tbfw` file instead.
6. Click **Flash**. The status LED on the board blinks quickly while the firmware is written. **Don't unplug the cable until the console says "Flash complete."**
7. The board restarts on its own. Check that the version number in the Firmware tab has changed.

> **If flashing fails partway through:** Hold the board's **BOOT** button, tap **RESET**, release **BOOT**, and click **Flash** again. This forces the board into recovery mode.

## Part 2: Calibrate the line sensor

Before tuning, the robot needs to learn what "black" and "white" look like under your room's lighting.

1. Open the **Sensors** tab. You'll see a live bar for each sensor in the array.
2. Place the robot on the track with the sensor array over the tape.
3. Click **Start sensor calibration**, then slowly slide the robot side to side across the line for about five seconds so that every sensor sees both the tape and the background.
4. Click **Finish**. The bars should now read close to 0 over white and close to 1000 over black.
5. Center the robot on the line and check that the **Line position** readout is near `0`. Moving the robot left should make the number negative, and moving it right should make it positive. If the signs are backwards, turn on **Invert sensor order**.

## Part 3: Tune the PID controller

The robot steers using a **PID controller**, which combines three terms:

- **P (proportional)** steers harder the farther the robot is from the line.
- **D (derivative)** reacts to how fast the error is changing. It damps wobble.
- **I (integral)** corrects small errors that build up over time. Line followers often need little or none.

Open the **Tuning** tab. Set a modest **Base speed**, around 40%, while you tune. Once the robot follows the line reliably, you can raise it.

Tune one value at a time:

1. **Set all gains to zero.** Then raise **Kp** in small steps, clicking **Apply** and running the robot each time. Stop when the robot follows the line but wobbles back and forth.
2. **Add Kd.** Increase it until the wobble settles down. Too much Kd makes the robot twitchy or jittery, so back off a little if that happens.
3. **Add Ki only if needed.** If the robot consistently drifts to one side on straight sections, add a very small Ki. If it starts to swing wider over time, reduce it.
4. **Test the curves.** If the robot runs off the outside of a curve, increase Kp slightly or lower the base speed.
5. **Raise the speed.** Increase base speed in 5% steps. Each step usually needs a little more Kd.

The **Live graph** panel plots line position over time. A good tune looks like a flat line with small bumps. Large regular waves mean too much P or too little D.

> **Tip:** Write down each set of values you try along with what happened. It's easy to lose track, and your notes will help the next person who tunes this robot.

## Part 4: Save a calibration profile

The gains you apply are stored only in memory until you save them. If you unplug the robot now, they'll be lost.

1. Click **Save profile** in the Tuning tab.
2. Give the profile a descriptive name, such as `club-track-fast` or `competition-mat-v2`.
3. Select **Write to board** so the robot loads this profile every time it starts up.
4. Also select **Save copy to laptop**. This creates a `.tbcal` file you can share with teammates or reload later from **Profiles > Import**.

The board can store up to four profiles. To switch between them without a laptop, hold the **MODE** button when the robot powers on. The LED blinks once for each profile slot.

## Wrapping up

You've flashed new firmware, calibrated the sensors, tuned the PID controller, and saved a profile. Recalibrate the sensors (Part 2) whenever you move to a new room or track surface. Different lighting and materials change the sensor readings, even if your PID values stay the same.
