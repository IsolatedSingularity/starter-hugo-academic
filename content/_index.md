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
    design:
      background:
        image:
          filename: main-background.jpg
          filters:
            brightness: 0.35
          size: cover
          position: center
          parallax: true
          text_color_light: true
      
  - block: collection
    id: publications
    content:
      title: Publications
      text: |
        My undergraduate research on the swampland criteria and de Sitter vacua in string theory, published in the McGill Science Undergraduate Research Journal.
      filters:
        folders:
          - publication
        exclude_featured: false
    design:
      columns: '2'
      view: citation
      background:
        image:
          filename: main-background.jpg
          filters:
            brightness: 0.35
          size: cover
          position: center
          parallax: true
          text_color_light: true
      
  - block: portfolio
    id: projects
    content:
      title: Research
      filters:
        folders:
          - project
        tags:
          - Research

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

  - block: portfolio
    id: creative
    content:
      title: Creative Work
      filters:
        folders:
          - project
        tags:
          - Creative

    design:
      columns: '2'
      view: showcase
      background:
        image:
          filename: creative-background.jpg
          filters:
            brightness: 0.6
          size: cover
          position: center
          parallax: true
          text_color_light: true

  - block: markdown
    content:
      title: ""
      text: |
        <div style="text-align: center; padding: 2rem 0; font-size: 0.85rem; color: #888;">
        Background artwork by <a href="https://www.tumblr.com/ionomycin" target="_blank" style="color: #41f0c1;">Ionomycin</a>, <a href="https://www.dominikmayer.art/" target="_blank" style="color: #41f0c1;">Dominik Mayer</a>, and <a href="https://www.tumblr.com/mintaii" target="_blank" style="color: #41f0c1;">mintaii</a>!
        </div>
        <div style="text-align: center; padding: 1rem 0; display: flex; justify-content: center;">
        <img src="/media/unknown.jpg" alt="" style="max-width: 120px; opacity: 0.8;">
        </div>
    design:
      columns: '1'
      

---
