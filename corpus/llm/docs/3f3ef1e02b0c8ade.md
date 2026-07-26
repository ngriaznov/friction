# Programming Channel Memories with ChirpLine

This guide walks you through loading a set of channel memories onto a handheld transceiver using ChirpLine, a desktop utility for building and uploading memory files. It assumes you already have a programming cable for your specific radio model and have installed ChirpLine on your computer.

## Before You Start

Gather the following:

- Your handheld transceiver, fully charged or on external power
- A USB programming cable rated for your radio's model number (cables are not universal — check the connector on your radio and the chipset in the cable)
- ChirpLine installed on a Windows, macOS, or Linux machine
- A list of the frequencies, offsets, and tones you want to program

Radios draw more current during a write cycle than during normal operation, so a low battery can cause a failed or partial upload. Plugging into wall power, or at least starting with a full charge, avoids this entirely.

## Step 1: Connect the Programming Cable

Turn the radio off before connecting anything. Plug the cable into the radio's data port — usually located under a rubber cover on the side, near the speaker/mic jack — then plug the USB end into your computer. Turn the radio back on and set it to whatever mode your manual calls "clone" or "PC programming" mode, if your model requires it explicitly. Most modern handhelds do not require this step and will respond to ChirpLine automatically.

## Step 2: Open ChirpLine and Select Your Radio

Launch ChirpLine and choose **Radio > Detect Model** from the menu. ChirpLine will query the radio over the serial connection and identify the make and model automatically. If detection fails, open **Radio > Select Model** and choose your transceiver manually from the list, along with the correct serial port (shown as a COM port on Windows or a /dev/tty device on macOS and Linux).

## Step 3: Build or Import Your Memory File

If you're starting from scratch, use the spreadsheet-style grid in ChirpLine's main window to enter each channel: name, receive frequency, transmit offset and direction, tone mode, and tone frequency. Each row becomes one memory slot.

If you already have a memory file — either one you built previously or one shared by your local club — choose **File > Open** and select it. ChirpLine reads its native `.cln` format as well as plain CSV exports from most other programming tools, so importing a club repeater list someone else prepared usually just works.

Review every row before uploading. Pay particular attention to duplex direction and offset value, since these are the two fields most commonly transposed by mistake.

## Step 4: Upload to the Radio

With the cable still connected and the memory file open, select **Radio > Upload to Radio**. ChirpLine will show a progress bar as it writes each memory slot. Do not disconnect the cable, remove power, or turn off the radio during this process — an interrupted upload can leave the radio's memory in an inconsistent state, occasionally requiring a factory reset to recover.

When the upload completes, ChirpLine displays a confirmation dialog. Disconnect the cable, power-cycle the radio, and step through a few of the new channels manually to confirm names and frequencies match what you intended.

## Step 5: Save a Backup

Before you close ChirpLine, use **File > Save As** to keep a copy of the memory file you just uploaded. Keeping dated backups makes it easy to restore your configuration after a firmware update or a factory reset, and to compare changes over time.
