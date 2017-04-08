# coding=utf-8
"""
Red lights, green lights, strawberry wine
A good friend of mine
Follows the stars,
Venus and Mars are alright tonight!

Produce a _nightly_ list of observable comets brighter than 16 mag
for the Robo-AO queue

@autor: Dr Dmitry A. Duev [Caltech]


"""

import datetime
from solarsyslib2 import TargetListComets, hms, dms
import pytz
import os
import inspect
import ConfigParser
import argparse
from astropy.time import Time

if __name__ == '__main__':
    # create parser
    parser = argparse.ArgumentParser(prog='comets_tonight.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Create nightly target list for the comets program.')
    # positional arguments
    parser.add_argument('config_file', type=str, help='config file')

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

    f_inp = config.get('Path', 'pypride_inp')

    # observatory and time zone
    observatory = config.get('Observatory', 'observatory')
    timezone = config.get('Observatory', 'timezone')

    # observability settings:
    # nighttime between twilights: astronomical (< -18 deg), civil (< -6 deg), nautical (< -12 deg)
    twilight = config.get('Comets', 'twilight')
    # fraction of night when observable given constraints:
    fraction = float(config.get('Comets', 'fraction'))
    # magnitude limit:
    m_lim = float(config.get('Comets', 'm_lim'))
    # elevation cut-off [deg]:
    elv_lim = float(config.get('Comets', 'elv_lim'))

    ''' target list [no limits on Vmag] '''
    # date in UTC!!! (for KP, it's the next day if it's still daytime)
    now = datetime.datetime.now(pytz.timezone(timezone))
    today = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)
    # print('\nstarted at:', datetime.datetime.now(pytz.timezone(timezone)))

    tl = TargetListComets(f_inp, database_source='mpc', database_file='CometEls.txt',
                          _observatory=observatory, _m_lim=m_lim, _elv_lim=elv_lim, _date=today)

    comet = [c for c in tl.database if c['name'] == '0041P'][0]
    # print(comet)
    mjd = Time.now().tdb.mjd
    radec, radec_dot, Vmag = tl.get_obs_params(comet, mjd, 'J2000', True)
    # print(mjd, radec, radec_dot, Vmag)

    radec = [hms(radec[0]), dms(radec[1])]
    ra_str = '{:02.0f}:{:02.0f}:{:06.3f}'.format(*radec[0])
    if radec[1][0] >= 0:
        dec_str = '{:02.0f}:{:02.0f}:{:06.3f}'.format(radec[1][0], abs(radec[1][1]), abs(radec[1][2]))
    else:
        dec_str = '{:03.0f}:{:02.0f}:{:06.3f}'.format(radec[1][0], abs(radec[1][1]), abs(radec[1][2]))

    print('0 {:s} {:s} {:s} {:s}'.format(ra_str, dec_str,
                                         '{:.5f}'.format(radec_dot[0]), '{:.5f}'.format(radec_dot[1])))
