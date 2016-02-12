from __future__ import print_function
import argparse
import os
import inspect
import ConfigParser

if __name__ == '__main__':
    # load config data
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(abs_path, 'config.ini'))

    # create parser
    parser = argparse.ArgumentParser(prog='fixtargnums.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Reset target numbering.')
    # optional arguments
    # filter targets by these criteria
    # parser.add_argument('-c', '--comment', action='store', type=str,
    #                     help='')
    # parser.add_argument('-n', '--name', action='store', type=str,
    #                     help='')
    # positional arguments
    parser.add_argument('program_number', type=str, help='object name')

    # a parser exception (e.g. if no argument was given) will be caught
    args = parser.parse_args()

    ''' change XML files '''
    path = config.get('Path', 'program_path')
    program_path = os.path.join(path, 'Program_{:d}'.format(int(args.program_number)))

    xml_list = [f for f in os.listdir(program_path) if 'Target' in f and f[0] != '.']

    for xml in xml_list:
        with open(os.path.join(program_path, xml), 'r') as f:
            f_lines = f.read()
        xml_targ_num = int(xml[xml.index('_')+1:xml.index('.xml')])
        target_number_str = f_lines[f_lines.index('<number>'):f_lines.index('</number>')+9]
        # replace only the first appearance (which is the target number):
        f_lines = f_lines.replace(target_number_str, '<number>{:d}</number>'.format(xml_targ_num), 1)
        with open(os.path.join(program_path, xml), 'w') as f:
            f.write(f_lines)
