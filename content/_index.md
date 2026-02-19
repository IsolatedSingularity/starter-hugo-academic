---
# Leave the homepage title empty to use the site title
title: Jeffrey Morais
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
        <em>My undergraduate research on the swampland criteria and de Sitter vacua in string theory, published in the McGill Science Undergraduate Research Journal.</em>
      filters:
        folders:
          - publication
        exclude_featured: false
    design:
      columns: '2'
      view: citation
      spacing:
        padding: ["60px", "0", "20px", "0"]
      background:
        image:
          filename: main-background.jpg
          filters:
            brightness: 0.25
          size: cover
          position: center
          parallax: true
          text_color_light: true
      
  - block: portfolio
    id: work
    content:
      title: Work
      text: |
        <em>Quantum software and post-quantum cryptographic tooling developed at <a href="https://www.btq.com/" target="_blank" style="color: #41f0c1;">BTQ Technologies</a>, spanning error correction, consensus protocols, random number generation, and industry analytics.</em>
      filters:
        folders:
          - project
        tags:
          - Work

    design:
      columns: '2'
      view: showcase
      spacing:
        padding: ["60px", "0", "20px", "0"]
      background:
        image:
          filename: ship.jpg
          filters:
            brightness: 0.7
          size: cover
          position: center
          parallax: true
          text_color_light: true

  - block: portfolio
    id: projects
    content:
      title: Research
      text: |
        <em>Theoretical and computational physics research conducted across <a href="https://www.mcgill.ca/" target="_blank" style="color: #41f0c1;">McGill University</a>, <a href="https://www.fudan.edu.cn/en/" target="_blank" style="color: #41f0c1;">Fudan University</a>, <a href="https://www.uvic.ca/" target="_blank" style="color: #41f0c1;">University of Victoria</a>, <a href="https://muhc.ca/" target="_blank" style="color: #41f0c1;">McGill University Health Centre</a>, <a href="https://www.ualberta.ca/" target="_blank" style="color: #41f0c1;">University of Alberta</a>, <a href="https://www.concordia.ca/" target="_blank" style="color: #41f0c1;">Concordia University</a>, and <a href="https://www.vaniercollege.qc.ca/" target="_blank" style="color: #41f0c1;">Vanier College</a>.</em>
      filters:
        folders:
          - project
        tags:
          - Research

    design:
      columns: '2'
      view: showcase
      spacing:
        padding: ["60px", "0", "20px", "0"]
      background:
        image:
          filename: background.jpg
          filters:
            brightness: 0.6
          size: cover
          position: center
          parallax: true
          text_color_light: true

  - block: portfolio
    id: creative
    content:
      title: Personal
      text: |
        <em>Something watches from the edges of the screen. You probably shouldn't scroll further.</em>
      filters:
        folders:
          - project
        tags:
          - Creative

    design:
      columns: '2'
      view: showcase
      spacing:
        padding: ["60px", "0", "20px", "0"]
      background:
        image:
          filename: aether.jpg
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
        Background artwork by <a href="https://www.tumblr.com/ionomycin" target="_blank" style="color: #41f0c1;">Ionomycin</a>, <a href="https://www.dominikmayer.art/" target="_blank" style="color: #41f0c1;">Dominik Mayer</a>, <a href="https://en.wikipedia.org/wiki/Montague_Dawson" target="_blank" style="color: #41f0c1;">Montague Dawson</a>, <a href="https://www.instagram.com/mo_ninglj/" target="_blank" style="color: #41f0c1;">moninlj</a>, <a href="https://www.pixiv.net/en/users/9678597" target="_blank" style="color: #41f0c1;">Y_Y</a>, and <a href="https://www.tumblr.com/mintaii" target="_blank" style="color: #41f0c1;">mintaii</a>!
        </div>
        <div style="text-align: center; padding: 1rem 0; display: flex; justify-content: center;">
        <img src="/media/unknown.jpg" alt="" style="max-width: 120px; opacity: 0.8;">
        </div>
    design:
      columns: '1'
      

---
