# coding=utf-8
"""
The Asteroids Galaxy Tour

Produce a _nightly_ list of observable asteroids brighter than 16 mag
for the Robo-AO queue

@autor: Dr Dmitry A. Duev [Caltech]


"""

from __future__ import print_function
import os
import numpy as np
import datetime
from astropy.table import Table
from astropy import units as u
from solarsyslib import TargetListAsteroids, TargetXML, \
                        get_guide_star, asteroid_multiples_numbers
import pytz
import ConfigParser
import inspect
import argparse


if __name__ == '__main__':
    # create parser
    parser = argparse.ArgumentParser(prog='asteroids_tonight.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Create nightly target list for the asteroids program.')
    # optional arguments
    parser.add_argument('-m', '--multiples', action='store_true',
                        help='process only known multiples')
    # positional arguments
    # parser.add_argument('name', type=str, help='object name')

    # a parser exception (e.g. if no argument was given) will be caught
    args = parser.parse_args()

    # load config data
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(abs_path, 'config.ini'))

    # asteroid database:
    path_to_database = config.get('Path', 'asteroid_database_path')
    f_database = os.path.join(path_to_database, 'ELEMENTS.NUMBR')

    f_inp = config.get('Path', 'pypride_inp')

    # process all or only known multiples?
    if args.multiples:
        ''' known main-belt multiples '''
        # triples:
        # 45 Eugenia, 87 Sylvia, 93 Minerva, 130 Elektra, 216 Kleopatra, 3749 Balam
        # all known multiples
        multiples_num = asteroid_multiples_numbers()
        # mask by asteroid number:
        mask = multiples_num - 1
    else:
        # get 'em all
        mask = None

    ''' target list [no limits on Vmag] '''
    # date in UTC!!! (for KP, it's the next day if it's still daytime)
    now = datetime.datetime.now(pytz.timezone("America/Phoenix"))
    today = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)

    tl = TargetListAsteroids(f_database, f_inp, _observatory='kitt peak', _m_lim=16.5, date=today)
    targets = tl.target_list_observable(tl.target_list_all(today, mask, parallel=True), today)

    ''' find guide stars for targets with Vmag>16.5 '''
    if 1 == 0:
        # target table for vizier:
        dim_targets = [t for t in targets if 16.5 < t[4] < 20.5]
        dim_targets_table = [(t[2][0]*180/np.pi, t[2][1]*180/np.pi, t[4]) for t in dim_targets]
        if len(dim_targets_table) > 0:
            dim_targets_table = Table(rows=dim_targets_table,
                                      names=('_RAJ2000', '_DEJ2000', 'Vmag'))
            # set units
            dim_targets_table['_RAJ2000'].unit = u.deg
            dim_targets_table['_DEJ2000'].unit = u.deg
            guide = get_guide_star(dim_targets_table, chunk=29, ang_sep='33s')

    ''' make/change XML files '''
    path = config.get('Path', 'program_path')
    program_number = config.get('Path', 'program_number_asteroids')

    txml = TargetXML(path=path, program_number=program_number,
                     server=config.get('Path', 'queue_server'))
    # dump 'em targets!
    c = txml.dumpTargets(targets, epoch='J2000')

    if c is None:
        print('Successfully updated the target list via the website')

        # clean up the target list - remove unobserved, which are not suitable anymore:
        txml.clean_target_list()

    # go for high priority asteroids:
    asteroid_hp_num = np.array(map(int, config.get('HP', 'asteroids_hp').split(',')))

    # mask by asteroid number:
    mask = asteroid_hp_num - 1

    tl = TargetListAsteroids(f_database, f_inp, _observatory='kitt peak', _m_lim=17.5, date=today)
    targets = tl.target_list_observable(tl.target_list_all(today, mask, parallel=True), today)

    ''' make/change XML files '''
    path = config.get('Path', 'program_path')
    program_number = config.get('Path', 'program_number_asteroids_hp')

    txml = TargetXML(path=path, program_number=program_number,
                     server=config.get('Path', 'queue_server'))
    # dump 'em targets!
    c = txml.dumpTargets(targets, epoch='J2000')

    if c is None:
        print('Successfully updated the target list via the website')

        # clean up the target list - remove unobserved, which are not suitable anymore:
        txml.clean_target_list()
