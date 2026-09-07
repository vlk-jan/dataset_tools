# Vendored front-end libraries

| File | What | Version | Licence |
| --- | --- | --- | --- |
| `leaflet.js`, `leaflet.css` | [Leaflet](https://leafletjs.com), the map the GNSS track is drawn on | 1.9.4 (unmodified, from `unpkg.com/leaflet@1.9.4/dist/`) | BSD-2-Clause, `LEAFLET-LICENSE` |

Vendored rather than loaded from a CDN because the viewer is routinely run on a
cluster node or inside a container with no route to the internet, and a map
whose library fails to load is a blank box with no explanation. The *tiles*
still need the network -- without it the track is drawn on a plain backdrop and
the map says so.

Leaflet's default marker icons (`images/marker-icon.png`, `layers.png`) are
deliberately **not** vendored: `map.js` draws every marker as a `divIcon`
styled in `style.css`, so nothing here asks for an image file.

To upgrade, replace both files from the same release, update the version above,
and check `map.js` against the release notes.
