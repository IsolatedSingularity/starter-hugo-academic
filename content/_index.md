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
        <div style="margin-bottom: 2.5rem;"><em>Early work on the swampland criteria and de Sitter vacua in string theory, published in the McGill Science Undergraduate Research Journal. Established the mathematical foundations that inform my approach to quantum software design.</em></div>
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
        <div style="margin-bottom: 2.5rem;"><em>Production quantum software and post-quantum cryptographic systems developed at <a href="https://www.btq.com/" target="_blank" style="color: #41f0c1;">BTQ Technologies</a>, from error correction toolkits and consensus protocol engineering to quantum random number generation and threat analytics.</em></div>
      filters:
        folders:
          - project
        tags:
          - Work

    design:
      columns: '2'
      view: showcase
      background:
        image:
          filename: ship.jpg
          filters:
            brightness: 0.5
          size: cover
          position: center
          parallax: true
          text_color_light: true

  - block: portfolio
    id: projects
    content:
      title: Research
      text: |
        <div style="margin-bottom: 2.5rem;"><em>Computational physics and quantum information research underpinning my engineering work, conducted across <a href="https://www.mcgill.ca/" target="_blank" style="color: #41f0c1;">McGill University</a>, <a href="https://www.fudan.edu.cn/en/" target="_blank" style="color: #41f0c1;">Fudan University</a>, <a href="https://www.uvic.ca/" target="_blank" style="color: #41f0c1;">University of Victoria</a>, <a href="https://muhc.ca/" target="_blank" style="color: #41f0c1;">McGill University Health Centre</a>, <a href="https://www.ualberta.ca/" target="_blank" style="color: #41f0c1;">University of Alberta</a>, <a href="https://www.concordia.ca/" target="_blank" style="color: #41f0c1;">Concordia University</a>, and <a href="https://www.vaniercollege.qc.ca/" target="_blank" style="color: #41f0c1;">Vanier College</a>.</em></div>
      filters:
        folders:
          - project
        tags:
          - Research

    design:
      columns: '2'
      view: showcase
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
        <div style="margin-bottom: 2.5rem;"><em>You feel it before you see it. A flicker at the edge of the viewport. It knows you scrolled this far. It has been waiting.</em></div>
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
