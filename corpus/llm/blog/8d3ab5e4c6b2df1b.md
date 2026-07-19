# Building a Chrome Extension to Block Autoplaying Videos: A Developer's Journey

Autoplaying videos can be incredibly disruptive, especially when browsing news websites where multiple videos might start playing simultaneously. As part of my ongoing efforts to improve user experience online, I decided to create a small Chrome extension that blocks autoplaying videos on news sites. This blog post will walk you through the development process, including the necessary permissions and one interesting bug we encountered.

## The Problem Statement

News websites often use autoplaying videos as a means to grab attention and increase engagement. However, this can lead to an overwhelming experience for users who find themselves bombarded by multiple videos playing simultaneously in their browser tabs. To address this issue, I decided to build a Chrome extension that would automatically block these autoplaying videos.

## Setting Up the Project

To start, I created a new directory for my project and set up a basic file structure:
```
autoplay-blocker/
│
├── manifest.json
├── background.js
├── content.js
└── popup.html
```

### `manifest.json`: The Heart of the Extension

The `manifest.json` file is crucial as it defines the extension's permissions, its metadata (like name and version), and how it interacts with different parts of Chrome. For our project, we needed to request the following permissions:

```json
{
  "name": "Autoplay Blocker",
  "version": "1.0",
  "description": "Blocks autoplaying videos on news sites.",
  "permissions": [
    "activeTab", 
    "tabs"
  ],
  "background": {
    "scripts": ["background.js"],
    "persistent": false
  },
  "content_scripts": [
    {
      "matches": ["*://*.news.com/*", "*://*.techsite.com/*"], // Replace with actual news site URLs
      "js": ["content.js"]
    }
  ],
  "manifest_version": 3
}
```

- **`activeTab` and `tabs`:** These permissions allow the extension to interact with active tabs.
- **`content_scripts`:** This section defines which web pages the content script (in our case, `content.js`) should run on. Replace `*://*.news.com/*` and `*://*.techsite.com/*` with the actual URLs of your target news sites.

## The Content Script: `content.js`

The content script is responsible for detecting and blocking autoplaying videos. Here's a basic implementation:

```javascript
// content.js

function blockAutoplayVideos() {
  const videoElements = document.querySelectorAll('video, iframe[src*="youtube.com"], iframe[src*="vimeo.com"]');
  
  videoElements.forEach(video => {
    if (video.autoplay) {
      // Temporarily pause the video to prevent it from starting
      video.pause();
      
      // Add a listener to ensure the video is not resumed by any script
      video.addEventListener('play', () => video.pause());
    }
  });
}

// Run the blocking function on page load
document.addEventListener('DOMContentLoaded', blockAutoplayVideos);
```

This script selects all `<video>` and iframe elements that have an autoplay attribute. It then pauses these videos to prevent them from playing automatically.

## A Curious Bug: Lazy-Loaded iFrames

During testing, I noticed a peculiar issue with certain news sites that used lazy-loaded iframes. The autoplay blocker would not work as expected on these pages. Upon investigation, it turned out that the iframes were loaded asynchronously using JavaScript after the initial DOM had been parsed.

To address this, I modified `content.js` to listen for iframe load events and apply the same blocking logic:

```javascript
function blockAutoplayVideos() {
  const videoElements = document.querySelectorAll('video, iframe[src*="youtube.com"], iframe[src*="vimeo.com"]');
  
  videoElements.forEach(video => {
    if (video.autoplay) {
      // Temporarily pause the video to prevent it from starting
      video.pause();
      
      // Add a listener to ensure the video is not resumed by any script
      video.addEventListener('play', () => video.pause());
    }
  });

  // Handle lazy-loaded iFrames
  document.addEventListener('DOMNodeInserted', (event) => {
    if (event.target.tagName === 'IFRAME') {
      const iframe = event.target;
      blockAutoplayVideosInIframe(iframe);
    }
  });
}

function blockAutoplayVideosInIframe(iframe) {
  new Promise((resolve, reject) => {
    iframe.onload = resolve;
    iframe.onerror = reject;
  }).then(() => {
    // Block autoplay in the iframe
    const iframeDocument = iframe.contentDocument || iframe.contentWindow.document;
    if (iframeDocument) {
      const videoElementsInIframe = iframeDocument.querySelectorAll('video, iframe[src*="youtube.com"], iframe[src*="vimeo.com"]');
      videoElementsInIframe.forEach(video => {
        if (video.autoplay) {
          video.pause();
          video.addEventListener('play', () => video.pause());
        }
      });
    }
  });
}

document.addEventListener('DOMContentLoaded', blockAutoplayVideos);
```

This updated script now listens for `DOMNodeInserted` events, ensuring that any newly inserted iFrames are also checked for autoplaying videos.

## Conclusion

Building a Chrome extension to block autoplaying videos on news sites was an interesting project that involved understanding both the basics of web development and the intricacies of browser APIs. The experience highlighted the importance of considering edge cases like lazy-loaded iframes, which can introduce unique challenges when developing extensions.

If you're interested in trying out this extension or exploring similar projects, feel free to clone the repository and test it on your own news sites!
