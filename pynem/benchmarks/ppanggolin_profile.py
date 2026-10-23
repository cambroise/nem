#!/usr/bin/env python3
"""
Profiles pynem's partition_pangenome (the ppanggolin/NEM path), on a
pre-generated dataset.
  --profiler pyinstrument (default): statistical sampling profiler
  --profiler cprofile: deterministic call-counting profiler
"""

import argparse
import cProfile
import io
import pstats
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

import networkx as nx
import numpy as np

from pynem import partition_pangenome

DEFAULT_OUT = {
    "html": "profile/report_pyinstrument.html",
    "console": None,
    "json": "profile/report.json",
    "speedscope": "profile/report.speedscope.json",
    "cprofile": "profile/report_cprofile.html",
}

CPROFILE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 24px; }}
  h1 {{ font-size: 18px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
  .flame {{ overflow-x: auto; border: 1px solid #ddd; border-radius: 6px; padding: 8px; background: #fff; }}
  .flame svg {{ display: block; width: 100%; height: auto; min-width: 900px; }}
  pre {{ background: #f6f6f6; border: 1px solid #ddd; border-radius: 6px; padding: 12px;
         overflow-x: auto; font-size: 12px; margin-top: 24px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">{meta}</div>
<div class="flame">
{svg}
</div>
<pre>{summary}</pre>
</body>
</html>
"""


def _pyinstrument_renderers():
    from pyinstrument.renderers import (
        ConsoleRenderer,
        HTMLRenderer,
        JSONRenderer,
        SpeedscopeRenderer,
    )

    return {
        "html": HTMLRenderer,
        "console": ConsoleRenderer,
        "json": JSONRenderer,
        "speedscope": SpeedscopeRenderer,
    }


def load(data_dir):
    presence_path = Path(data_dir) / "presence.csv"
    graph_path = Path(data_dir) / "graph.txt"
    X = np.loadtxt(presence_path, delimiter=",")

    edges = []
    max_node = -1
    with open(graph_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            i, j = int(parts[0]), int(parts[1])
            w = float(parts[2]) if len(parts) > 2 else 1.0
            edges.append((i, j, w))
            max_node = max(max_node, i, j)

    G = nx.DiGraph()
    G.add_nodes_from(range(max(max_node + 1, X.shape[0])))
    for (i, j, w) in edges:
        G.add_edge(i, j, weight=w)
    return X, G


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--sm-degree", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--profiler", choices=["pyinstrument", "cprofile"], default="pyinstrument")
    ap.add_argument("--renderer", choices=["html", "console", "json", "speedscope"],
                     default="html", help="pyinstrument only; ignored with --profiler cprofile")
    ap.add_argument("--out", type=str, default=None,
                     help="output file")
    ap.add_argument("--interval", type=float, default=1e-3,
                     help="pyinstrument sampling interval in seconds, default 1ms")
    ap.add_argument("--show-all", action="store_true",
                     help="by default pyinstrument collapses any frame whose file "
                          "lives under the venv site-packages")
    ap.add_argument("--top", type=int, default=40,
                     help="cprofile only; number of functions in the cumulative-time summary")
    ap.add_argument("--keep-pstats", type=str, default=None,
                     help="cprofile only; also dump the raw .pstats file to this path")
    ap.add_argument("--no-warmup", action="store_true",
                     help="skip the untimed warm-up call that normally excludes numba "
                          "JIT compilation cost from the profile")
    args = ap.parse_args()

    out_path = args.out or DEFAULT_OUT[args.profiler if args.profiler == "cprofile" else args.renderer]

    X, G = load(args.data_dir)
    print(f"N={X.shape[0]} D={X.shape[1]} edges={G.number_of_edges()}", file=sys.stderr)

    if not args.no_warmup:
        partition_pangenome(X, G, K=args.k, beta=args.beta, sm_degree=args.sm_degree,
                             max_iter=1, tol=args.tol)

    if args.profiler == "cprofile":
        profiler = cProfile.Profile()
        profiler.enable()
        result = partition_pangenome(X, G, K=args.k, beta=args.beta, sm_degree=args.sm_degree,
                                      max_iter=args.max_iter, tol=args.tol)
        profiler.disable()

        print(f"n_iter={result['n_iter']}", file=sys.stderr)

        stats = pstats.Stats(profiler, stream=sys.stderr)
        stats.sort_stats("cumulative")
        stats.print_stats()

        buf = io.StringIO()
        buf_stats = pstats.Stats(profiler, stream=buf)
        buf_stats.sort_stats("cumulative")
        buf_stats.print_stats(args.top)

        with tempfile.NamedTemporaryFile(suffix=".pstats") as tmp:
            stats.dump_stats(tmp.name)

            if args.keep_pstats:
                Path(args.keep_pstats).parent.mkdir(parents=True, exist_ok=True)
                Path(args.keep_pstats).write_bytes(Path(tmp.name).read_bytes())
                print(f"wrote {args.keep_pstats}", file=sys.stderr)

            svg = subprocess.run(["uvx", "flameprof", tmp.name],
                                  check=True, capture_output=True, text=True).stdout

        svg = svg.replace('<svg version="1.1"', '<svg version="1.1" preserveAspectRatio="xMinYMin meet"', 1)

        html = CPROFILE_HTML.format(
            title=escape(f"cProfile — {Path(args.data_dir).name}"),
            meta=escape(f"N={X.shape[0]} D={X.shape[1]} edges={G.number_of_edges()} "
                        f"n_iter={result['n_iter']} k={args.k} beta={args.beta}"),
            svg=svg,
            summary=escape(buf.getvalue()),
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(html)
        print(f"wrote {out_path}", file=sys.stderr)
        return

    renderers = _pyinstrument_renderers()
    from pyinstrument import Profiler

    profiler = Profiler(interval=args.interval)
    profiler.start()
    result = partition_pangenome(X, G, K=args.k, beta=args.beta, sm_degree=args.sm_degree,
                                  max_iter=args.max_iter, tol=args.tol)
    profiler.stop()

    print(f"n_iter={result['n_iter']}", file=sys.stderr)

    renderer_kwargs = {"show_all": True} if args.show_all and args.renderer != "html" else {}
    renderer = renderers[args.renderer](**renderer_kwargs)
    output = profiler.output(renderer=renderer)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(output)
        print(f"wrote {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
