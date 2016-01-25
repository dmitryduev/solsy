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
import xml.etree.ElementTree as ET
import requests


def clean_target_list(_path, _program_number, _server='http://localhost:8080'):
    target_xml_path = os.path.join(_path, 'Program_{:s}'.format(str(_program_number)))
    pnot = len([_f for _f in os.listdir(target_xml_path)
                    if 'Target_' in _f and _f[0] != '.'])
    # iterate over target xml files
    target_nums_to_remove = []

    target_list_xml = ['Target_{:d}.xml'.format(i+1) for i in range(int(pnot))]

    for targ_num, target_xml in enumerate(target_list_xml):
        tree = ET.parse(os.path.join(target_xml_path, target_xml))
        root = tree.getroot()

        targ = {}
        targ['Object'] = []
        for content in root:
            if content.tag != 'Object':
                targ[content.tag] = content.text
            else:
                obj = {}
                obj['Observation'] = []
                for data_obj in content:
                    if data_obj.tag != 'Observation':
                        obj[data_obj.tag] = data_obj.text
                    else:
                        obs = {}
                        for data_obs in data_obj:
                            obs[data_obs.tag] = data_obs.text
                        obj['Observation'].append(obs)
                targ['Object'].append(obj)

        t_xml = datetime.datetime.strptime(' '.join(targ['comment'].split()[-2:]),
                                           '%Y-%m-%d %H:%M:%S.%f')
        # updated > 2 days ago?
        if (datetime.datetime.now() - t_xml).total_seconds() > 86400*2 \
                and targ['done'] == '0':
            # print(targ['comment'], targ['done'])
            target_nums_to_remove.append(targ_num+1)

    # TODO: add try/except clause + check server status (should be 200 - OK)
    # now remove the xml files. start from end not to scoop numbering
    if len(target_nums_to_remove) > 0:
        try:
            for _targ_num in target_nums_to_remove[::-1]:
                r = requests.get(os.path.join(_server, 'removeTarget'),
                                 auth=('admin', 'robo@0'),
                                 params={'program_number': int(_program_number),
                                         'target_number': int(_targ_num)})
                # print(_targ_num, r.status_code)
                if int(r.status_code) != 200:
                    print('server error')
            print('removed {:d} targets that are no longer suitable for observations.'
                  .format(len(target_nums_to_remove)))
            # call main page to modify/fix Programs.xml
            r = requests.get(_server, auth=('admin', 'robo@0'))
            if int(r.status_code) != 200:
                    print('server error')
        except Exception:
            print('failed to remove targets via the website.')


if __name__ == '__main__':
    # asteroid database:
    path_to_database = '/Users/dmitryduev/_caltech/roboao/asteroids/'
    f_database = os.path.join(path_to_database, 'ELEMENTS.numbr')

    f_inp = '/Users/dmitryduev/_jive/pypride/src/pypride/inp.cfg'

    # process all or only known multiples?
    do = '_all_'

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
    today = datetime.datetime(now.year, now.month, now.day)

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
    txml = TargetXML(path=path, program_number=program_number)
    # dump 'em targets!
    txml.dumpTargets(targets, epoch='J2000')

    # clean up the target list - remove unobserved, which are not suitable anymore:
    txml.clean_target_list(_server='http://localhost:8080')
