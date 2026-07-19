Title: Building a Small Chrome Extension to Block Autoplaying Videos on News Sites

As an avid news reader, I've often found myself frustrated by the constant autoplaying videos on news websites. Not only can they be annoying, but they also consume bandwidth and battery life. In this blog post, we'll explore how to build a small Chrome extension that blocks autoplaying videos on news sites.

**Getting Started**

To create a Chrome extension, you'll need to have some basic knowledge of HTML, CSS, JavaScript, and the Chrome Extension API. If you're new to Chrome extensions, I recommend checking out the official documentation for more information.

First, create a new directory for your project and create the following files:

* `manifest.json`: This file contains metadata about your extension, such as its name, description, and permissions.
* `popup.html`: This file will contain the HTML for your popup that appears when you click the extension's icon in the browser toolbar.
* `script.js`: This file will contain the JavaScript code that does the heavy lifting of blocking autoplaying videos.

**Manifest Permissions**

To access certain features on a webpage, such as playing or pausing video content, we need to request specific permissions in our `manifest.json` file. In this case, we'll need the following permissions:

* `"activeTab"`: This permission allows us to access the currently active tab.
* `"scripting"`: This permission allows us to inject scripts into webpages.

Here's an example of what our `manifest.json` file might look like:
```json
{
  "name": "Autoplay Blocker",
  "version": "1.0",
  "manifest_version": 2,
  "description": "A Chrome extension that blocks autoplaying videos on news sites.",
  "permissions": ["activeTab", "scripting"],
  "browser_action": {
    "default_popup": "popup.html"
  }
}
```
**Blocking Autoplaying Videos**

In our `script.js` file, we'll use the `contentScript` API to inject a script into webpages that will block autoplaying videos. We'll also use the `activeTab` permission to access the currently active tab.

Here's an example of what our `script.js` file might look like:
```javascript
function blockAutoplay() {
  // Get the currently active tab
  chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
    var tab = tabs[0];
    
    // Check if the webpage has a video element
    var videoElements = document.querySelectorAll('video');
    if (videoElements.length > 0) {
      // Get the first video element
      var videoElement = videoElements[0];
      
      // Add an event listener to pause the video when it's clicked
      videoElement.addEventListener('click', function() {
        this.pause();
      });
      
      // Add an event listener to prevent autoplaying videos
      videoElement.addEventListener('play', function() {
        this.pause();
      }, { once: true });
    }
  });
}

// Listen for the "load" event on the popup HTML file
document.addEventListener("DOMContentLoaded", blockAutoplay);
```
**The Bug**

One issue we encountered was that the script only worked on webpages with a video element in the same document. However, some news sites use lazy-loaded iframes to load their content. In these cases, our script wouldn't be able to access the iframe's video elements.

To fix this bug, we needed to modify our `script.js` file to also check for video elements within iframes. We can do this by using the `contentScript` API's `runAt` option to run our script in the context of all webpages, including those with lazy-loaded iframes.

Here's an updated version of our `manifest.json` file:
```json
{
  "name": "Autoplay Blocker",
  "version": "1.0",
  "manifest_version": 2,
  "description": "A Chrome extension that blocks autoplaying videos on news sites.",
  "permissions": ["activeTab", "scripting"],
  "content_scripts": [
    {
      "matches": ["*://*/*"],
      "run_at": "document_end",
      "js": ["script.js"]
    }
  ],
  "browser_action": {
    "default_popup": "popup.html"
  }
}
```
With these changes, our extension should now work on all webpages, including those with lazy-loaded iframes.

**Conclusion**

In this blog post, we explored how to build a small Chrome extension that blocks autoplaying videos on news sites. We covered the manifest permissions needed and one bug that only showed up on sites with lazy-loaded iframes. With these changes, our extension should now work on all webpages, providing users with a more enjoyable browsing experience.
