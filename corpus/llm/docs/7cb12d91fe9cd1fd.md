**Thistle Static Site Generator Configuration File Format**

The configuration file for Thistle is a critical component that defines how your static site will be generated and structured. It allows you to set global settings, specify the theme, output directory, and more. Below, we detail the supported top-level keys in the configuration file, explain the syntax for per-page front matter overrides, and provide two complete example config files.

### Supported Top-Level Keys

1. **title** (Required): This key sets the title of your entire website or project. It appears on various pages as a header or subtitle, providing a consistent branding element across all pages.

   ```json
   "title": "My Personal Site"
   ```

2. **base_url** (Optional but recommended): Specifies the base URL where your site will be hosted. This is particularly useful for relative links in navigation and asset paths.

   ```json
   "base_url": "https://example.com/"
   ```

3. **theme**: Defines the path to the theme directory that Thistle should use for styling and layout purposes. The theme contains templates, stylesheets, and other assets necessary for rendering your pages.

   ```json
   "theme": "./themes/custom-theme"
   ```

4. **output_dir** (Optional): Indicates the directory where the generated static files will be placed after the build process completes. If not specified, Thistle defaults to a `dist` folder in the project root.

   ```json
   "output_dir": "./public"
   ```

### Syntax for Per-Page Front Matter Overrides

Thistle supports YAML front matter at the beginning of each markdown file to override any global settings defined in the configuration file. This is particularly useful for individual pages that require different titles, layouts, or other attributes.

Front matter should be placed between triple dashes (`---`) and can include keys such as `title`, `layout`, or custom variables specific to your site's needs. For example:

```yaml
---
title: "Welcome to My Blog"
layout: blog_post_layout
author: John Doe
---
```

In this example, the page titled "Welcome to My Blog" will use a different layout (`blog_post_layout`) and include an author attribution that might not be set globally in the config file.

### Example Config Files

#### Basic Configuration (config.json)

```json
{
  "title": "Thistle Documentation",
  "base_url": "https://thistle-static-site.com/",
  "theme": "./themes/default-theme",
  "output_dir": "./build"
}
```

This example configuration sets a basic structure for a Thistle-generated site, providing the necessary information to host it at a specific URL, using a default theme, and outputting files into a `build` directory.

#### Advanced Configuration with Custom Theme (config_advanced.json)

```json
{
  "title": "My Portfolio",
  "base_url": "https://johnsmithportfolio.com/",
  "theme": "./themes/custom-portfolio-theme",
  "output_dir": "./static-content",

  "posts_per_page": 5,
  "author_info": {
    "name": "John Smith",
    "bio": "I am a web developer and designer based in New York."
  }
}
```

This advanced configuration file includes additional settings such as `posts_per_page` to control pagination on the blog section, and an `author_info` block within front matter for each post. It also specifies a custom theme directory, demonstrating how you can tailor both global site attributes and specific page behaviors.

### Conclusion

The Thistle configuration file serves as the backbone for defining your static site's structure, appearance, and functionality. By understanding and utilizing its supported top-level keys, along with leveraging per-page front matter overrides, you can create a highly customized and efficient website that meets your unique requirements. The provided examples illustrate how to start with basic settings and expand into more complex configurations as needed.
