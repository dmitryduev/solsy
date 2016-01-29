# coding=utf-8
"""
The Asteroids Galaxy Tour

Produce a _nightly_ list of observable asteroids brighter than 16 mag
for the Robo-AO queue

@autor: Dr Dmitry A. Duev [Caltech]


"""

import os
import numpy as np
import datetime
from astropy.table import Table
from astropy import units as u
from solarsyslib import TargetListAsteroids, TargetXML, get_guide_star
import pytz


if __name__ == '__main__':
    # asteroid database:
    path_to_database = '/Users/dmitryduev/_caltech/roboao/asteroids/'
    f_database = os.path.join(path_to_database, 'ELEMENTS.NUMBR')

    f_inp = '/Users/dmitryduev/_jive/pypride/src/pypride/inp.cfg'

    # process all or only known multiples?
    do = 'all'

    if do != 'all':
        ''' known main-belt multiples '''
        # triples:
        # 45 Eugenia, 87 Sylvia, 93 Minerva, 130 Elektra, 216 Kleopatra, 3749 Balam
        # all known multiples
        with open('/Users/dmitryduev/_caltech/roboao/asteroids/multiples.txt') as f:
            f_lines = f.readlines()
        multiples_num = np.array([int(l.split()[0].replace('(', '').replace(')', ''))
                                  for l in f_lines])
        # mask by asteroid number:
        mask = multiples_num - 1
    else:
        # get 'em all
        mask = None

    ''' target list [no limits on Vmag] '''
    # date in UTC!!! (for KP, it's the next day if it's still daytime)
    now = datetime.datetime.now(pytz.timezone("America/Phoenix"))
    today = datetime.datetime(now.year, now.month, now.day+1)

    tl = TargetListAsteroids(f_database, f_inp, _observatory='kitt peak', _m_lim=16)
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
    path = '/Users/dmitryduev/web/qserv/operation'
    program_number = 4
    txml = TargetXML(path=path, program_number=program_number, server='http://localhost:8081')
    # dump 'em targets!
    txml.dumpTargets(targets, epoch='J2000')

    # clean up the target list - remove unobserved, which are not suitable anymore:
    txml.clean_target_list()
