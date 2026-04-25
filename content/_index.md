---
# Leave the homepage title empty to use the site title
title:
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Welcome, Traveler.
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
    design:
      background:
        image:
          filename: bio.png
          filters:
            brightness: 0.525
          size: cover
          position: center
          parallax: true
          text_color_light: true
      
  - block: portfolio
    id: work
    content:
      title: Work
      text: |
        <div style="margin-bottom: 2.5rem;"><em>Production quantum software and post-quantum cryptographic systems developed at <a href="https://www.btq.com/" target="_blank" style="color: #d4607a;">BTQ Technologies</a>, from error correction toolkits and consensus protocol engineering to quantum random number generation and threat analytics.</em></div>
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
          filename: work.png
          filters:
            brightness: 0.544
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
          filename: ship.jpg
          filters:
            brightness: 0.495
          size: cover
          position: center
          parallax: true
          text_color_light: true

  - block: portfolio
    id: projects
    content:
      title: Research
      text: |
        <div style="margin-bottom: 2.5rem;"><em>Computational physics and quantum information research that underpins my engineering work, spanning quantum many-body thermalization, topological neural networks, holographic entanglement, and signal processing.</em></div>
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
          filename: research.jpg
          filters:
            brightness: 0.64
          size: cover
          position: center
          parallax: true
          text_color_light: true

  - block: markdown
    content:
      title: ""
      text: |
        <div style="text-align: center; padding: 1.5rem 0 0.5rem 0; font-size: 0.85rem; color: #888;">
        Friends: <a href="https://beau-coup.github.io/" target="_blank" style="color: #d4607a; text-decoration: none;">&gt;_ Alex</a>,&nbsp; <a href="https://maryiletey.com/" target="_blank" style="color: #d4607a; text-decoration: none;">~~ Mary</a>,&nbsp; <a href="https://guillaumepayeur.github.io/" target="_blank" style="color: #d4607a; text-decoration: none;">// Guillaume</a>,&nbsp; <a href="https://thomasrribeiro.com/" target="_blank" style="color: #d4607a; text-decoration: none;">{} Thomas</a>,&nbsp; and <a href="https://rioweil.github.io/" target="_blank" style="color: #d4607a; text-decoration: none;">(+) Ryohei</a>
        </div>
        <div style="text-align: center; padding: 2rem 0; font-size: 0.85rem; color: #888;">
        Background artwork by <a href="https://www.tumblr.com/ionomycin" target="_blank" style="color: #d4607a;">Ionomycin</a>, <a href="https://www.dominikmayer.art/" target="_blank" style="color: #d4607a;">Dominik Mayer</a>, <a href="https://en.wikipedia.org/wiki/Montague_Dawson" target="_blank" style="color: #d4607a;">Montague Dawson</a>, <a href="https://www.instagram.com/mo_ninglj/" target="_blank" style="color: #d4607a;">moninlj</a>, <a href="https://www.pixiv.net/en/users/9678597" target="_blank" style="color: #d4607a;">Y_Y</a>, and <a href="https://www.tumblr.com/mintaii" target="_blank" style="color: #d4607a;">mintaii</a>!
        </div>
        <div style="text-align: center; padding: 1rem 0; display: flex; justify-content: center;">
        <img src="/media/unknown.jpg" alt="" style="max-width: 120px; opacity: 0.8;">
        </div>
    design:
      columns: '1'
      

---
