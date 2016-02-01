# coding=utf-8
"""
Red lights, green lights, strawberry wine
A good friend of mine
Follows the stars,
Venus and Mars are alright tonight!

Produce a _nightly_ list of observable planets/moons brighter than 16 mag
for the Robo-AO queue

@autor: Dr Dmitry A. Duev [Caltech]


"""

import datetime
from solarsyslib import TargetListPlanetsAndMoons, TargetXML
import pytz
import os
import inspect
import ConfigParser

if __name__ == '__main__':
    # load config data
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(abs_path, 'config.ini'))

    f_inp = config.get('Path', 'pypride_inp')

    ''' target list [no limits on Vmag] '''
    # date in UTC!!! (for KP, it's the next day if it's still daytime)
    now = datetime.datetime.now(pytz.timezone("America/Phoenix"))
    today = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(days=1)

    tl = TargetListPlanetsAndMoons(f_inp, _observatory='kitt peak', _m_lim=16)
    # tl.target_list_all(today)
    targets = tl.target_list_observable(tl.target_list_all(today), today,
                                        fraction=0.05)

    ''' make/change XML files '''
    path = config.get('Path', 'program_path')
    program_number = config.get('Path', 'program_number_planets')
    txml = TargetXML(path=path, program_number=program_number,
                     server=config.get('Path', 'queue_server'))
    # dump 'em targets!
    txml.dumpTargets(targets, epoch='J2000')
    print('Succesfully updated the target list via the website')
