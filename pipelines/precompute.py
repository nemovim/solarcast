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
    parser.add_argument('--hard-exit-after-success', dest='hard_exit_after_success', action='store_true', default=True, help='Bypass native shutdown hooks after successful GPU processing (default)')
    parser.add_argument('--normal-exit', dest='hard_exit_after_success', action='store_false', help='Use normal Python shutdown')
    args = parser.parse_args()
    if args.bounds and args.tile_index:
        parser.error('Choose either --bounds or --tile-index')
    common = ['--device', args.device]
    if not args.hard_exit_after_success:
        common.append('--normal-exit')
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
