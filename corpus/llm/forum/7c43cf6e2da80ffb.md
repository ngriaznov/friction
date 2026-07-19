### How to Design a Command-Line Tool in Python for Deduplicating Large Photo Libraries Using Perceptual Hashing

#### Question:
I'm working on a project where I need to deduplicate a large photo library using a command-line tool in Python. I've heard that perceptual hashing is more effective than byte-level checksums, but I'm not sure how to implement it. Can you guide me through the process?

---

### Answer:

Deduplicating a large photo library can be challenging due to the sheer volume of images and the need for high accuracy in identifying near-duplicates. Perceptual hashing is an excellent choice because it focuses on the visual content rather than the raw byte data, making it more robust against variations like compression artifacts or minor changes.

#### Why Use Perceptual Hashing?

Perceptual hashing algorithms (like pHash and dHash) generate a hash value that represents the perceptually important features of an image. This means that even if two images are slightly different (e.g., due to lighting, resolution, or compression), their hashes will still be similar enough to indicate they might be duplicates.

In contrast, byte-level checksums like MD5 or SHA-256 compare every single byte in the file, which can lead to false negatives for visually identical but not byte-for-byte identical images. For example, if an image is saved and re-saved with minor changes (like a slight rotation), its byte content will differ significantly from the original.

#### Step 1: Choose a Perceptual Hashing Algorithm

For this task, we'll use `pHash`, which is widely used for image deduplication. It works by converting an image into a spectrogram and then computing the discrete cosine transform (DCT) to extract features that are invariant to scaling, rotation, and minor distortions.

You can install `pHash` using pip:
```bash
pip install pHash
```

#### Step 2: Implementing Perceptual Hashing

First, let's create a function to generate perceptual hashes for images:

```python
from PIL import Image
import phash

def get_phash(image_path):
    image = Image.open(image_path)
    return phash.hash_image(image)
```

This function opens an image and returns its perceptual hash.

#### Step 3: Bucketing Near-Duplicate Hashes with Hamming Distance

To handle near-duplicates, we'll use the Hamming distance to measure how similar two hashes are. The Hamming distance between two strings of equal length is the number of positions at which the corresponding symbols are different.

We can define a threshold for what constitutes a "near-duplicate." For example, if the Hamming distance is less than 5 out of 32 (the size of a typical perceptual hash), we consider them near-duplicates:

```python
def are_near_duplicates(hash1, hash2):
    return phash.hamming_distance(hash1, hash2) < 5
```

#### Step 4: Handling Burst-Mode Shots

Burst-mode shots often result in images that are visually very similar but not exact duplicates. To handle these cases effectively:

1. **Group by Time**: First, group photos by the time they were taken. This can help reduce the number of comparisons needed.
2. **Use a Sliding Window**: Implement a sliding window approach where you compare each image against its neighbors within a certain time frame (e.g., 30 seconds).
3. **Refine with Visual Similarity**: For images that are close in time, use perceptual hashing to determine if they are near-duplicates.

Here’s how you might implement this:

```python
from datetime import timedelta

def group_by_time(images):
    # Group by time and then sort within each group
    return {t: sorted(g) for t, g in itertools.groupby(sorted(images), key=lambda x: x['timestamp'])}

def deduplicate_burst_mode(images, time_window=timedelta(seconds=30)):
    grouped_images = group_by_time(images)
    
    duplicates = set()
    for timestamp, images_in_group in grouped_images.items():
        if len(images_in_group) > 1:
            current_image_hash = get_phash(images_in_group[0]['path'])
            for image in images_in_group[1:]:
                if are_near_duplicates(current_image_hash, get_phash(image['path'])):
                    duplicates.add(image)
    
    return list(duplicates)
```

#### Step 5: Command-Line Interface

Finally, let's create a simple command-line interface to run the deduplication tool:

```python
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Deduplicate photo library using perceptual hashing.")
    parser.add_argument("directory", type=str, help="Path to the directory containing photos")
    args = parser.parse_args()

    images = [f for f in Path(args.directory).glob("**/*") if f.is_file() and f.suffix.lower() in ['.jpg', '.png']]
    
    duplicates = deduplicate_burst_mode(images)
    print(f"Found {len(duplicates)} potential duplicates.")

if __name__ == "__main__":
    main()
```

This script will scan the specified directory, group images by time, and identify near-duplicates using perceptual hashing.

---

By following these steps, you can create a robust command-line tool for deduplicating large photo libraries. Perceptual hashing ensures that visually similar but not byte-identical images are correctly identified as duplicates.
