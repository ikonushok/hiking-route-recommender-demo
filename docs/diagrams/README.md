# Diagrams

Interactive architecture diagrams rendered with [Archify](https://github.com/tt-a1i/archify).

| File | Type | Live |
|---|---|---|
| `monitoring-stack.architecture.json` | source specification | — |
| `monitoring-stack.html` | delivered artifact | [open](https://ikonushok.github.io/hiking-route-recommender-demo/diagrams/monitoring-stack.html) |

The HTML is self-contained: inline SVG, light/dark themes, pan and zoom, relationship
tracing, presentation mode and PNG/SVG export. GitHub does not render HTML files inside
the repository, so use the live link above (served by GitHub Pages from `/docs`).

## Regenerate

```bash
npx skills add tt-a1i/archify -g
node ~/.agents/skills/archify/bin/archify.mjs validate architecture \
    docs/diagrams/monitoring-stack.architecture.json --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver architecture \
    docs/diagrams/monitoring-stack.architecture.json \
    docs/diagrams/monitoring-stack.html --quality showcase --json
```

Edit the JSON specification, never the generated HTML.
