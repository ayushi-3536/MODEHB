import os
import argparse

def create_output_dir(args):
    # Directory where files will be written
    if args.folder is None:
        folder = "dehb"
        if args.version is not None:
            folder = "dehb_v{}".format(args.version)
    else:
        folder = args.folder

    output_path = os.path.join(args.output_path, folder)
    return output_path
