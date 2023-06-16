---
# Leave the homepage title empty to use the site title
title: Jeff
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Hello There!
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
    

---
