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
      text: |
        <div style="margin-top: -0.5rem; margin-bottom: 0.5rem; font-size: 1.05rem; color: #aaa;">Quantum Software Engineer</div>
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

  - block: markdown
    id: skills
    content:
      title: ""
      text: |
        <div style="text-align: center; padding: 1.5rem 0;">
          <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.6rem; max-width: 700px; margin: 0 auto;">
            <span style="background: rgba(65, 240, 193, 0.1); border: 1px solid rgba(65, 240, 193, 0.3); color: #41f0c1; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem;">Python</span>
            <span style="background: rgba(65, 240, 193, 0.1); border: 1px solid rgba(65, 240, 193, 0.3); color: #41f0c1; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem;">Qiskit</span>
            <span style="background: rgba(65, 240, 193, 0.1); border: 1px solid rgba(65, 240, 193, 0.3); color: #41f0c1; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem;">NumPy / SciPy</span>
            <span style="background: rgba(65, 240, 193, 0.1); border: 1px solid rgba(65, 240, 193, 0.3); color: #41f0c1; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem;">TypeScript</span>
            <span style="background: rgba(65, 240, 193, 0.1); border: 1px solid rgba(65, 240, 193, 0.3); color: #41f0c1; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem;">Next.js</span>
            <span style="background: rgba(65, 240, 193, 0.1); border: 1px solid rgba(65, 240, 193, 0.3); color: #41f0c1; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem;">Git</span>
            <span style="background: rgba(65, 240, 193, 0.1); border: 1px solid rgba(65, 240, 193, 0.3); color: #41f0c1; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem;">Linux</span>
            <span style="background: rgba(65, 240, 193, 0.1); border: 1px solid rgba(65, 240, 193, 0.3); color: #41f0c1; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem;">Quantum Error Correction</span>
            <span style="background: rgba(65, 240, 193, 0.1); border: 1px solid rgba(65, 240, 193, 0.3); color: #41f0c1; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem;">Post-Quantum Cryptography</span>
          </div>
        </div>
    design:
      columns: '1'
      background:
        color: '#000000'

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
            brightness: 0.3
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
        <div style="margin-bottom: 2.5rem;"><em>Creative projects and explorations in game development, procedural generation, and fluid dynamics. You feel it before you see it. A flicker at the edge of the viewport. It knows you scrolled this far.</em></div>
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
