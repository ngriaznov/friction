# lcd-i2c-font

An Arduino library for 16x2 character LCDs on an I2C backpack, with support for custom glyphs beyond the display's built-in character ROM.

HD44780-compatible displays come with a fixed character set and only eight user-definable slots (CGRAM). lcd-i2c-font manages those slots for you. You can define as many custom glyphs as you want, including arrows, battery icons, accented letters and small pictograms, and the library loads them into CGRAM when you print them. You no longer have to decide which eight symbols a sketch gets.

## Features

- Drives 16x2 (and 20x4) HD44780 displays through a PCF8574 I2C backpack
- A built-in extra glyph set: arrows, battery levels, signal bars, degree/thermometer, bell, lock, heart, and common accented Latin letters
- Define your own 5x8 glyphs as byte arrays
- Automatic CGRAM slot management, so glyphs load on demand and the least recently used slot is reused
- A familiar `print()` interface, since the class extends `Print`

## Installation

### Arduino Library Manager (recommended)

1. Open the Arduino IDE.
2. Go to **Sketch → Include Library → Manage Libraries…**
3. Search for **lcd-i2c-font**.
4. Click **Install**.

The library depends only on `Wire`, which comes with every Arduino core.

### arduino-cli

```sh
arduino-cli lib install "lcd-i2c-font"
```

### PlatformIO

```ini
lib_deps = example/lcd-i2c-font
```

## Wiring

The I2C backpack reduces the LCD's 16 pins to four:

| Backpack | Arduino Uno / Nano | ESP32 | ESP8266 (NodeMCU) |
|---|---|---|---|
| GND | GND | GND | GND |
| VCC | 5V | 5V (VIN) | VIN |
| SDA | A4 | GPIO 21 | D2 |
| SCL | A5 | GPIO 22 | D1 |

Some things to know:

- **I2C address.** Most backpacks use `0x27`. Boards with the PCF8574A chip use `0x3F`. If you're not sure, run the `I2CScanner` example that ships with this library.
- **Contrast.** If the backlight is on but nothing appears, turn the blue potentiometer on the backpack. A new module often arrives set to show a row of solid blocks, or nothing at all.
- **Power.** The LCD needs 5V. On 3.3V boards such as the ESP32, power the backpack from 5V. Most backpacks include pull-ups on SDA and SCL, so the bus sits at 5V. That is usually fine with ESP32s in practice, but a level shifter is the safe choice.
- **Long wires.** Keep I2C runs under about 50 cm. For longer runs, lower the bus speed with `lcd.setClock(50000)`.

## Quick start

```cpp
#include <Wire.h>
#include <LcdI2cFont.h>

LcdI2cFont lcd(0x27, 16, 2);   // address, columns, rows

void setup() {
  lcd.begin();
  lcd.backlight();

  lcd.setCursor(0, 0);
  lcd.print("Temp: 21.5");
  lcd.printGlyph(GLYPH_DEGREE);
  lcd.print("C");

  lcd.setCursor(0, 1);
  lcd.printGlyph(GLYPH_BATTERY_75);
  lcd.print(" 78%  ");
  lcd.printGlyph(GLYPH_HEART);
}

void loop() {}
```

`printGlyph()` checks whether the glyph is already in CGRAM. If it is, it writes that slot's character code. If it isn't, it uploads the bitmap into a free slot first, or into the least recently used slot when all eight are taken.

## Defining a custom glyph

A glyph is a 5x8 bitmap stored as eight bytes, one per row from top to bottom. Only the lower five bits of each byte are used, and bit 4 is the leftmost pixel.

```cpp
// A small house
const uint8_t HOUSE[8] PROGMEM = {
  0b00100,
  0b01110,
  0b11111,
  0b10001,
  0b10101,
  0b10101,
  0b11111,
  0b00000
};
```

Register it once in `setup()` to get a glyph ID, then print it the same way as a built-in glyph:

```cpp
GlyphId houseGlyph;

void setup() {
  lcd.begin();
  houseGlyph = lcd.defineGlyph(HOUSE);
  lcd.printGlyph(houseGlyph);
  lcd.print(" Home");
}
```

`defineGlyph()` only stores a pointer to your bitmap. Nothing is sent to the display until the glyph is printed. Keeping bitmaps in `PROGMEM` saves RAM on AVR boards; on ESP32 and ESP8266 it's harmless.

If you need to change a glyph while the sketch runs, for example for a simple animation, call `lcd.updateGlyph(id, newBitmap)`. When the glyph is already on screen, its CGRAM slot is rewritten, and every place it appears changes at once.

A browser-based editor for drawing bitmaps and copying the resulting array is included in `extras/glyph-editor.html`.

## The eight-slot limit

The hardware limit still exists: **no more than eight different custom glyphs can be on screen at the same time.** If you print a ninth, the least recently used slot is reused, and any cell on screen still showing the old glyph will change to the new one. Call `lcd.glyphsInUse()` to see how many slots are currently taken. Plan screens so that each one needs eight distinct custom glyphs or fewer.

## Examples

These are available under **File → Examples → lcd-i2c-font**:

- `HelloGlyphs`: prints every built-in glyph, one page at a time
- `CustomGlyph`: defines and prints your own bitmap
- `BatteryMeter`: an animated battery indicator using `updateGlyph()`
- `I2CScanner`: finds your backpack's address

## License

MIT. See `LICENSE`.
