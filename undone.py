from __future__ import print_function
import argparse
import os
import inspect
import ConfigParser

if __name__ == '__main__':

    # create parser
    parser = argparse.ArgumentParser(prog='undone.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Reset done tags in target xml files.')
    # optional arguments
    # filter targets by these criteria
    parser.add_argument('-c', '--comment', action='store', type=str,
                        help='look for string in comment')
    parser.add_argument('-n', '--name', action='store', type=str,
                        help='look for string in source name')
    parser.add_argument('-d', '--done', action='store_true', type=str,
                        help='mark done instead')
    # positional arguments
    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)
    parser.add_argument('program_number', type=str, help='object name')

    # a parser exception (e.g. if no argument was given) will be caught
    args = parser.parse_args()
    config_file = args.config_file

    # load config data
    config = ConfigParser.RawConfigParser()
    try:
        config.read(config_file)
    except IOError:
        try:
            abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
            config.read(os.path.join(abs_path, 'config.ini'))
        except IOError:
            raise Exception('failed to read in the config file')

    ''' change XML files '''
    path = config.get('Path', 'program_path')
    program_path = os.path.join(path, 'Program_{:d}'.format(int(args.program_number)))

    xml_list = [f for f in os.listdir(program_path) if 'Target' in f and f[0] != '.']

    for xml in xml_list:
        changed = False
        with open(os.path.join(program_path, xml), 'r') as f:
            f_lines = f.read()
        # print(os.path.join(program_path, xml), f_lines)
        if (not args.comment and not args.name) or (args.comment and args.comment in f_lines) or \
                (args.name and args.name in f_lines):
            if not args.done:
                f_lines = f_lines.replace('<done>1</done>', '<done>0</done>')
            else:
                f_lines = f_lines.replace('<done>0</done>', '<done>1</done>')
            # print(os.path.join(program_path, xml), f_lines)
            # print(os.path.join(program_path, xml), '<done>1</done>' in f_lines)
            changed = True
        if changed:
            with open(os.path.join(program_path, xml), 'w') as f:
                f.write(f_lines)
