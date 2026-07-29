# zhihchenggao.github.io

Personal academic website of **Jingbo (Richard) Gao** — Computer Science undergraduate at the University of Chicago, research assistant at the Virginia Image and Video Analysis (VIVA) Lab.

Live at **[zhihchenggao.github.io](https://zhihchenggao.github.io)**.

## Built with

[al-folio](https://github.com/alshedivat/al-folio) v1.x, a thin [Jekyll](https://jekyllrb.com/) starter whose runtime ships as versioned plugin gems (`al_folio_core` and friends, pinned in the `Gemfile`). Deployed to GitHub Pages by `.github/workflows/deploy.yml` on every push to `main`.

## Local development

Requires Docker:

```bash
docker compose up -d          # serves at http://127.0.0.1:8080
docker compose logs -f        # build output
docker compose down
```

Content edits reload on refresh; `_config.yml` changes restart Jekyll automatically.

## Customizations

Because al-folio v1.x keeps its runtime in gems, local files that shadow a gem path override it. This site has a few:

| Path | Purpose |
| --- | --- |
| `_sass/_variables.scss` | UChicago palette — maroon accent, greystone neutrals |
| `_sass/_themes.scss` | Verbatim gem copy; exists so the file above resolves (Sass resolves `@use` relative to the importing file first) |
| `_sass/_custom.scss` | Typography, home profile column, CV page styling |
| `_layouts/about.liquid` | Moves the social icons under the headshot |
| `_layouts/cv.liquid` | Adds the resume link and obfuscates the email at render time |
| `assets/css/main.scss` | Gem copy plus one line loading `_custom.scss` |

Fonts (Source Serif 4, Source Sans 3, JetBrains Mono) are requested in `_config.yml` under `third_party_libraries.google_fonts` and applied in `_sass/_custom.scss` — both are required, since loading a font does not apply it.

`test/style_contract.js` has been narrowed: upstream forbids `_sass/`, `_layouts/`, and `_includes/`, which is correct for the al-folio starter repo but not for a site built from it.

## Content

- `_pages/` — about, CV, publications, projects
- `_data/cv.yml` — CV content (RenderCV format; also the input to the `render-cv` workflow, so `email` must stay a valid address)
- `_bibliography/papers.bib` — publications
- `assets/pdf/resume.pdf` — resume linked from the CV page

## License

Site content © Jingbo Gao. The al-folio theme is MIT licensed — see `LICENSE`.
