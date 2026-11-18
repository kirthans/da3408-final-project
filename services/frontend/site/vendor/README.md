# Locally vendored dependencies
Downloaded once from the npm registry; no runtime CDN requests.

- chess.js **1.4.0**, BSD-2-Clause: https://registry.npmjs.org/chess.js/-/chess.js-1.4.0.tgz
- jQuery **3.7.1**, MIT: https://registry.npmjs.org/jquery/-/jquery-3.7.1.tgz
- @chrisoakman/chessboardjs **1.0.0**, MIT: https://registry.npmjs.org/@chrisoakman/chessboardjs/-/chessboardjs-1.0.0.tgz

Each dependency directory includes its upstream LICENSE. The npm tarball omits images; the matching upstream v1.0.0 release (https://codeload.github.com/oakmac/chessboardjs/tar.gz/refs/tags/v1.0.0) supplies the Wikipedia chess pieces in `img/chesspieces/wikipedia/`, by Colin M. L. Burnett (Cburnett), distributed under the GNU Free Documentation License, CC-BY-SA 3.0 and GPL; attribution and license links: https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces . These PNGs are unchanged from the matching pinned upstream release.

- Mermaid **11.17.2**, MIT: https://registry.npmjs.org/mermaid/-/mermaid-11.17.2.tgz
  (self-contained browser bundle copied from the installed code-modernization plugin;
  embedded getVersion() identifies 11.17.2). Includes its upstream LICENSE.
