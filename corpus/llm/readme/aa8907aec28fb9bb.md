**Lumen: A Static Site Generator for Photography Portfolios**

Lumen is a lightweight, Ruby-based static site generator specifically designed for creating beautiful photography portfolios. With its simple folder convention and minimal configuration requirements, Lumen makes it easy to create stunning online galleries that showcase your best work.

**Folder Convention**

To get started with Lumen, you'll need to organize your content in the following way:

* Each gallery should have its own `images` directory, containing all the images for that particular gallery.
* Each page (e.g. about, contact, etc.) should be represented by a Markdown file without front-matter (i.e. no YAML metadata).
* The root of your project should contain a single YAML configuration file (`config.yml`) and any other static assets you want to include in your site.

Here's an example of what the folder structure might look like:
```markdown
project/
|--- config.yml
|--- images/
|    |--- gallery1/
|    |    |--- image1.jpg
|    |    |--- image2.jpg
|    |--- gallery2/
|    |    |--- image3.jpg
|    |    |--- image4.jpg
|--- pages/
|    |--- about.md
|    |--- contact.md
```
**Build Command**

To build your site, simply run the following command in your terminal:
```bash
bundle exec lumen build
```
This will generate a static HTML version of your site in the `output` directory.

**Local Preview Server**

If you want to preview your site locally before deploying it, you can use Lumen's built-in server:
```bash
bundle exec lumen serve
```
This will start a development server at `http://localhost:4000`, where you can view and interact with your site.

**Thumbnail Generation**

Lumen includes a simple thumbnail generation feature that automatically creates small versions of each image in the `images` directory. These thumbnails are then used throughout the site to improve performance and make navigation easier.

To customize the thumbnail sizes, simply add the following configuration options to your `config.yml` file:
```yml
thumbnail_sizes:
  - width: 400
    height: 300
  - width: 200
    height: 150
```
**EXIF-Based Caption Extraction**

Lumen can automatically extract captions from image EXIF data, making it easy to add metadata to your images. To enable this feature, simply add the following configuration option to your `config.yml` file:
```yml
extract_exif_captions: true
```
This will cause Lumen to read the caption field from each image's EXIF data and use it as the caption for that image.

**Configuration**

Lumen uses a single YAML file (`config.yml`) to store all configuration options. Here are some of the available settings:

* `site_title`: The title of your site, displayed in the browser tab.
* `theme`: The name of the theme you want to use (see below for more information on themes).
* `output_directory`: The directory where Lumen will generate the static HTML version of your site.

Here's an example `config.yml` file:
```yml
site_title: My Photography Portfolio
theme: default
output_directory: output
```
**Themes**

Lumen comes with a built-in `default` theme, but you can easily create and use custom themes by creating a new directory in the `themes` directory of your project. For example, to create a custom theme called `my-theme`, simply create a new directory `themes/my-theme` containing the necessary HTML templates.

To switch between themes, simply update the `theme` setting in your `config.yml` file:
```yml
theme: my-theme
```
**Conclusion**

Lumen is a lightweight, easy-to-use static site generator specifically designed for photography portfolios. With its simple folder convention and minimal configuration requirements, Lumen makes it easy to create stunning online galleries that showcase your best work. Whether you're a professional photographer or just starting out, Lumen is the perfect tool for creating a beautiful online portfolio.

**Getting Started**

To get started with Lumen, simply add the following line to your `Gemfile`:
```ruby
gem 'lumen'
```
Then run `bundle install` and follow the instructions above to build and deploy your site.
