# Archived CV page

The rendered `/cv/` page was retired in favor of linking the Resume tab straight to `assets/pdf/resume.pdf` (see `_pages/cv.html`). Jekyll ignores this
folder, so nothing here is built.

To restore the old page, delete `_pages/cv.html` and move the files back:

| Archived file     | Original location                 |
| ----------------- | --------------------------------- |
| `cv.md`           | `_pages/cv.md`                    |
| `cv.liquid`       | `_layouts/cv.liquid`              |
| `cv.yml`          | `_data/cv.yml`                    |
| `rendercv/*.yaml` | `assets/rendercv/`                |
| `render-cv.yml`   | `.github/workflows/render-cv.yml` |
