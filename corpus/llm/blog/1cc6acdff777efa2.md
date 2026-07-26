# What I Learned Writing My First Rust CLI

I had about four thousand photos with names like `IMG_4471.JPG` sitting in a folder, and I wanted them renamed to `2019-08-14_163022.jpg` based on the EXIF timestamp. This is a twenty-line Python script. I wrote it in Rust instead, because I had been reading about Rust for a year and had written none of it.

The tool is called `exifname`. It walks a directory, reads the `DateTimeOriginal` tag out of each file, and renames accordingly, with a `--dry-run` flag because I did not trust myself. It took a weekend. Python would have taken an hour. I don't regret it.

## The parts that went fine

Crate selection was easy. `clap` for arguments, `walkdir` for recursion, `kamadak-exif` for the metadata. The derive macro in `clap` is genuinely delightful — you write a struct with the fields you want and get a parser, help text, and validation for free.

Error handling clicked faster than I expected. I started by unwrapping everything, then swapped in `anyhow::Result` and the `?` operator, and suddenly every function that could fail said so in its signature. Coming from Python, where any line can raise anything, this felt less like a restriction and more like someone had turned the lights on.

## The parts that did not

Three things about the borrow checker confused me, and they were not the things I had been warned about.

**Iterating while mutating.** My first draft collected filenames into a `Vec<String>`, then looped over it, and inside the loop pushed conflicts onto the same vector. Obvious in hindsight. The compiler told me I couldn't borrow `names` as mutable while it was borrowed as immutable, and I spent twenty minutes trying to sprinkle `&` and `clone()` around the problem before understanding that the problem was my design. The fix was to collect conflicts into a second vector and merge afterward. I have written this bug in Python and shipped it.

**Strings.** I did not understand why I had `String`, `&str`, `Path`, `PathBuf`, `OsString`, and `&OsStr` all in play at once, and why the compiler would not let me hand one to a function expecting another. It took me embarrassingly long to internalize the pattern: the owned type when you're storing it, the borrowed type when you're passing it. Once that landed, `&Path` in arguments and `PathBuf` in structs became automatic. The `OsString` distinction is about filenames that aren't valid UTF-8, which I had never once thought about in fifteen years of scripting.

**Closures capturing things.** I wanted to filter the walker with a closure that referenced a config struct, and got a lifetime error I could not read. I fixed it by cloning the two fields I actually needed into the closure. That is probably not the elegant answer, but it compiled and it was correct, and I've made peace with the fact that my first Rust program contains a `clone()` I don't strictly need.

## Worth it?

The finished binary is 2.1 MB, starts instantly, and processed all four thousand photos in under two seconds. But that's not why it was worth it.

It was worth it because the compiler forced me to answer questions I normally defer: what happens if the file has no EXIF data, what happens if two photos have the same timestamp, what happens if the target name already exists. In Python I would have found out in production, on my own photo library. Here I found out at compile time, in the form of a `match` arm I hadn't filled in.

Next one will be faster.
