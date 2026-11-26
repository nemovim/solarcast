"""Precompute panel and roof results with the selected model weights."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', default='best_model', help='Checkpoint filename in models/panel')
    parser.add_argument('--roof', default='model', help='SAM ViT-H checkpoint filename in models/roof')
    parser.add_argument('--panel-config', help='Config compatible with the panel checkpoint')
    parser.add_argument('--tile-index', help='CSV with x,y tile coordinates')
    parser.add_argument('--bounds', type=int, nargs=4, metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.bounds and args.tile_index:
        parser.error('Choose either --bounds or --tile-index')
    common = ['--device', args.device]
    if args.tile_index:
        common += ['--tile-index', args.tile_index]
    if args.bounds:
        common += ['--bounds', *(str(value) for value in args.bounds)]
    panel = [sys.executable, '-m', 'pipelines.precompute_panels', args.panel, *common]
    if args.panel_config:
        panel += ['--config', args.panel_config]
    subprocess.run(panel, check=True)
    subprocess.run([sys.executable, '-m', 'pipelines.precompute_roofs', args.roof, *common], check=True)


if __name__ == '__main__':
    main()
