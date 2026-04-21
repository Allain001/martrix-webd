# MatrixVis Deployment Checklist

## Before Publish

- Run `start_fullstack.bat` once on a clean machine and confirm the site opens.
- Run `powershell -ExecutionPolicy Bypass -File .\run_smoke_test.ps1`.
- Confirm the homepage, `?view=defense`, and at least one case-link URL all load.
- Confirm matrix state and case state can be restored from the shared URL.
- Confirm `favicon.svg`, `site.webmanifest`, and `og-card.svg` are reachable.

## Content Review

- Check homepage wording for public users, not only judges or teammates.
- Check case descriptions for teaching clarity and answer-defense narration.
- Check the defense mode wording so it sounds polished on a projector.
- Check all visible Chinese copy for final phrasing before public launch.

## Deploy

- Push `one_page_demo` to a Git repository.
- Deploy with Docker or `render.yaml`.
- After the service is live, open the public URL and run the same smoke checks manually.
- Test the copied share link on another browser or device.

## Final Presentation Prep

- Prepare one public-mode link for judges to browse.
- Prepare one defense-mode link with a preloaded case for live presentation.
- Keep one fallback local launch path with `start_fullstack.bat` in case the venue network is unstable.
