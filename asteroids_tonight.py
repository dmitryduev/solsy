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
from solarsyslib import TargetListAsteroids, TargetXML, get_guide_star, Site, geodetic2Ecef
import pytz
import ConfigParser
import inspect
import argparse


if __name__ == '__main__':
    # create parser
    parser = argparse.ArgumentParser(prog='asteroids_tonight.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Create nightly target list for the asteroids program.')
    # positional arguments
    parser.add_argument('config_file', type=str, help='config file')
    # optional arguments
    parser.add_argument('-m', '--multiples', action='store_true',
                        help='process only known multiples')

    # a parser exception (e.g. if no argument was given) will be caught
    args = parser.parse_args()

    # load config data
    config = ConfigParser.RawConfigParser()
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    try:
        if not os.path.isabs(args.config_file):
            config.read(os.path.join(abs_path, args.config_file))
        else:
            config.read(args.config_file)
    except IOError:
        print('failed to load config file {:s}, trying ./config.ini'.format(args.config_file))
        try:
            config.read(os.path.join(abs_path, 'config.ini'))
        except IOError:
            raise Exception('config file ./config.ini not found')

    # asteroid database:
    path_to_database = config.get('Path', 'asteroid_database_path')
    f_database = os.path.join(path_to_database, 'ELEMENTS.NUMBR')

    f_inp = config.get('Path', 'pypride_inp')

    # observatory and time zone
    observatory = config.get('Observatory', 'observatory')
    timezone = config.get('Observatory', 'timezone')

    # site: KPNO 2.1 m: 31.958182 N, -111.598273 W, 2086.7 m (31°57'29.5"N 111°35'53.8"W)
    r_GTRS = geodetic2Ecef(lat=31.958182 * np.pi / 180.0, lon=-111.598273 * np.pi / 180.0, h=2086.7e-3)
    r_GTRS = np.array(r_GTRS) * 1e3  # in meters
    kitt_peak = Site(r_GTRS)
    # print(kitt_peak.r_GTRS)
    # kitt_peak = None

    # observability settings:
    # nighttime between twilights: astronomical (< -18 deg), civil (< -6 deg), nautical (< -12 deg)
    twilight = config.get('Asteroids', 'twilight')
    # fraction of night when observable given constraints:
    fraction = float(config.get('Asteroids', 'fraction'))
    # magnitude limit:
    m_lim = float(config.get('Asteroids', 'm_lim'))
    # elevation cut-off [deg]:
    elv_lim = float(config.get('Asteroids', 'elv_lim'))

    # process all or only known multiples?
    if args.multiples:
        ''' known main-belt multiples '''
        # triples:
        # 45 Eugenia, 87 Sylvia, 93 Minerva, 130 Elektra, 216 Kleopatra, 3749 Balam
        # all known multiples
        multiples_num = np.array([45, 93])
        # mask by asteroid number:
        mask = multiples_num - 1
    else:
        # get 'em all
        mask = None

    ''' target list [no limits on Vmag] '''
    # date in UTC!!! (for KP, it's the next day if it's still daytime)
    now = datetime.datetime.now(pytz.timezone(timezone))
    today = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)

    tl = TargetListAsteroids(f_database, f_inp, _observatory=observatory, _m_lim=m_lim, date=today)
    targets = tl.target_list_observable(tl.target_list_all(today, mask, parallel=True,
                                                           epoch='J2000', station=kitt_peak,
                                                           output_Vmag=True), today,
                                        elv_lim=elv_lim, twilight=twilight, fraction=fraction)

    ''' find guide stars for targets with Vmag>16.5 '''
    if 1 == 0:
        # target table for vizier:
        dim_targets = [t for t in targets if m_lim < t[4] < 20.5]
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

    tl = TargetListAsteroids(f_database, f_inp, _observatory='kitt peak', _m_lim=16.5, date=today)
    targets = tl.target_list_observable(tl.target_list_all(today, mask, parallel=True,
                                                           epoch='J2000', station=kitt_peak,
                                                           output_Vmag=True), today)

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
