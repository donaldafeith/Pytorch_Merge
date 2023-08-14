# PyTorch Merge
# Copyright (C) 2023 Donalda Feith
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

## IO libraries
import argparse
import os
import shlex
import sys

## AI libraries
import torch
from transformers import AutoConfig

## Time estimation / progress bar
from tqdm.auto import tqdm

## Aux functions
def is_file(filepath):
    '''Checks if a single path is an actual file that does exist'''
    if not os.path.isfile(filepath):
        msg = "{0} is not a file".format(filepath)
        raise argparse.ArgumentTypeError(msg)
    else:
        return filepath

#def is_fileslist(fileslist):
#    '''Checks if a list of paths are actual files that do exist'''
#    for filepath in fileslist:
#        if not os.path.isfile(filepath):
#            msg = "{0} is not a file".format(filepath)
#            raise argparse.ArgumentTypeError(msg)
#        else:
#            return fileslist

def fullpath(relpath):
    '''Relative path to absolute'''
    if (type(relpath) is object or hasattr(relpath, 'read')): # relpath is either an object or file-like, try to get its name
        relpath = relpath.name
    return os.path.abspath(os.path.expanduser(relpath))

## Main function
def main(argv=None):
    '''Script entry point, can be used in commandline or as a Python module'''
    # Allow to be used as a module or in commandline, by storing the commandline arguments in function argument argv if empty
    if argv is None: # if argv is empty, fetch from the commandline
        argv = sys.argv[1:]
    elif isinstance(argv, str): # else if argv is supplied but it's a simple string, we need to parse it to a list of arguments before handing to argparse or any other argument parser
        argv = shlex.split(argv) # Parse string just like argv using shlex

    # Setup arguments parser
    parser = argparse.ArgumentParser(
        prog='PyTorch Merge',
        description='Merge LLM weights files that are split into multiple parts')

    parser.add_argument('-c', '--config', metavar='config.json',
                        type=is_file,
                        required=True,
                        help='The configuration file for the model architecture you are working with (usually config.json).')
    parser.add_argument('-b', '--bin', metavar='weights1.bin weights2.bin',
                        type=is_file,
                        required=True,
                        nargs='+',
                        help='The model\'s weights .bin files you want to merge, you can merge as many files as you need.')
    parser.add_argument('-o', '--output', metavar='output.bin',
                        type=str,
                        required=True,
                        help='Output filepath with the weights merged.')

    # Parse arguments (either from commandline or function argument when used as a module)
    args = parser.parse_args(argv)

    # Prompt user for input paths
    config_path = fullpath(args.config)
    input_paths = [fullpath(x) for x in args.bin]
    output_path = fullpath(args.output)

    # Load the configuration for the model you are working with
    config = AutoConfig.from_pretrained(config_path)

    print("Merging models, please wait...")
    # Load the first input model
    model_1 = torch.load(input_paths[0])
    # Make a copy that will be the merged output model
    output_model = model_1.copy()
    # Close first model to free up memory
    del model_1
    # Loop to merge each model
    for i in tqdm(range(1, len(input_paths)), "MODEL"):
        # Load the next input model .bin file
        model_i = torch.load(input_paths[i])

        # Merge the state dictionaries of the two models
        for key, value in tqdm(model_i.items(), desc="ITEM"):
            if key not in output_model:
                output_model[key] = value
            else:
                # If the keys already exist in the first model, average the values
                output_model[key] = (output_model[key] + value) / 2

        # Close model to free up memory
        del model_i

    # Save the merged model
    print("Done merging, saving merged model to disk, please wait (this may take a while)...")
    torch.save(output_model, output_path)

    print("Merged model saved in %s . Exiting." % output_path)
