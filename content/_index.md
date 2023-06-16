---
# Leave the homepage title empty to use the site title
title: Jeff
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Welcome Traveler.
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
    favicon: favicon.ico
      
  - block: portfolio
    id: projects
    content:
      title: Research
      filters:
        folders:
          - project
        author:
          - project
        link: ""

    design:
      # Choose how many columns the section has. Valid values: '1' or '2'.
      columns: '2'
      view: showcase
      background:
        image:
          # Name of image in `assets/media/`.
          filename: background.jpg
          # Apply image filters?
          filters:
            # Darken the image? Range 0-1 where 1 is transparent and 0 is opaque.
            brightness: 0.6
          #  Image fit. Options are `cover` (default), `contain`, or `actual` size.
          size: cover
          # Image focal point. Options include `left`, `center` (default), or `right`.
          position: center
          # Use a fun parallax-like fixed background effect on desktop? true/false
          parallax: true
          # Text color (true=light, false=dark, or remove for the dynamic theme color).
          text_color_light: true
      

---
