# lcd-i2c-font

An Arduino library for 16x2 character LCDs driven over I2C, with support for a custom font set that goes beyond the eight user-definable characters the HD44780 controller gives you.

The HD44780 has a fixed character ROM plus eight slots of CGRAM that you can fill with your own 5x8 bitmaps. Eight is rarely enough once you want degree signs, arrows, battery icons, accented letters and a couple of progress-bar segments. lcd-i2c-font keeps a larger glyph table in flash memory and swaps glyphs into CGRAM on demand as you print, so from your sketch's point of view you can have dozens of custom characters. It only ever loads the ones currently on screen, and it tells you when you have asked for more than eight distinct custom glyphs at once.

## Installation

Open the Arduino IDE, go to **Sketch > Include Library > Manage Libraries...**, search for `lcd-i2c-font` and click Install. The library depends on `Wire`, which ships with the IDE.

With the Arduino CLI:

```sh
arduino-cli lib install lcd-i2c-font
```

PlatformIO users can add `lcd-i2c-font` to `lib_deps` in `platformio.ini`.

## Wiring

The library expects the common PCF8574-based I2C backpack soldered to the 16-pin LCD header. Connect the backpack to your board like this:

| Backpack pin | Uno / Nano | Mega | ESP32 | ESP8266 |
|---|---|---|---|---|
| GND | GND | GND | GND | GND |
| VCC | 5V | 5V | 5V (see note) | 5V (see note) |
| SDA | A4 | 20 | GPIO 21 | D2 (GPIO 4) |
| SCL | A5 | 21 | GPIO 22 | D1 (GPIO 5) |

Notes:

- Most backpacks and LCDs want 5 V for the backlight and contrast to work properly. On 3.3 V boards such as the ESP32, power the backpack from 5 V; the PCF8574 will happily accept 3.3 V logic on SDA and SCL. If the display shows only black boxes or nothing at all, turn the blue contrast trimmer on the backpack first.
- The default I2C address is `0x27`. Some boards use `0x3F`. If nothing appears, run the I2C scanner example in **File > Examples > lcd-i2c-font > scanner** to find the address. The three solder jumpers A0 to A2 on the backpack change it.
- Keep I2C wires short (under 30 cm) or add 4.7 kΩ pull-ups if your board does not have them.

## Example: initialize the display and print a custom glyph

```cpp
#include <Wire.h>
#include <LcdI2cFont.h>

// address, columns, rows
LcdI2cFont lcd(0x27, 16, 2);

void setup() {
  Wire.begin();
  lcd.begin();
  lcd.backlight();

  // Load the bundled extended glyph set (arrows, degree sign, battery icons...)
  lcd.useFont(LCDFONT_EXTENDED);

  lcd.setCursor(0, 0);
  lcd.print("Temp: 23");
  lcd.printGlyph(GLYPH_DEGREE);
  lcd.print("C");

  lcd.setCursor(0, 1);
  lcd.printGlyph(GLYPH_BATTERY_75);
  lcd.print(" ");
  lcd.printGlyph(GLYPH_ARROW_UP);
  lcd.print(" 1.2 km");
}

void loop() {}
```

`printGlyph()` takes a glyph ID from the active font. The library checks whether that glyph is already in one of the eight CGRAM slots; if not, it loads it into a free slot (or the least recently used one) and then writes the slot's character code to the display. `print()` continues to work with ordinary ASCII text, and you can mix both freely.

You can also write glyphs inline with the `\x` escape and a slot number, but `printGlyph()` is preferred because it manages slot allocation for you.

If more than eight distinct custom glyphs are visible at once, `printGlyph()` returns `false` and draws a `?` instead. Redrawing the screen usually resolves this since eviction is per-screen, but if you genuinely need nine at once you will need to redesign the layout.

## Defining and uploading a custom character

Each glyph is a 5x8 bitmap stored as eight bytes, one per row, with the low five bits used. Draw it on paper or with an online HD44780 character designer, then write it out:

```cpp
const uint8_t heartGlyph[8] PROGMEM = {
  0b00000,
  0b01010,
  0b11111,
  0b11111,
  0b01110,
  0b00100,
  0b00000,
  0b00000
};
```

Register it with the library and it gets an ID you can pass to `printGlyph()`:

```cpp
uint8_t GLYPH_HEART;

void setup() {
  // ...
  GLYPH_HEART = lcd.defineGlyph(heartGlyph);
  lcd.printGlyph(GLYPH_HEART);
}
```

`defineGlyph()` stores a pointer to the bitmap, so the array must stay in scope for the life of the sketch. Marking it `PROGMEM` keeps it out of RAM on AVR boards; the library reads it with `pgm_read_byte` when it needs to load the slot.

To build a whole font rather than adding glyphs one at a time, create a table and pass it to `useFont()`:

```cpp
const LcdGlyph myFont[] PROGMEM = {
  { heartGlyph },
  { smileyGlyph },
  { skullGlyph },
};

lcd.useFont(myFont, 3);
```

Glyph IDs are then the indices into your table (0, 1, 2). The bundled `LCDFONT_EXTENDED` set is defined exactly the same way in `src/fonts/extended.cpp` if you want a starting point to copy.

If you would rather write straight to CGRAM yourself, `lcd.createChar(slot, bitmap)` behaves like the standard LiquidCrystal call, but be aware that the glyph manager will evict from that slot if it runs out of space.

## Compatibility

Tested on AVR (Uno, Nano, Mega), ESP8266, ESP32, RP2040 and SAMD21 boards. Any board with a working `Wire` implementation should be fine. Displays: 16x2 and 20x4 HD44780-compatible modules with PCF8574 or PCF8574A backpacks.

## License

MIT.
