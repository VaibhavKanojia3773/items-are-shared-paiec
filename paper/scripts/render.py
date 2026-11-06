import sys, pymupdf
d = pymupdf.open(sys.argv[1])
pages = [int(p) for p in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
for p in pages:
    d[p].get_pixmap(dpi=int(sys.argv[4]) if len(sys.argv) > 4 else 170).save(sys.argv[2].replace(".png", f"_{p}.png"))
