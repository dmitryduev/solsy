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


if __name__ == '__main__':
    f_inp = '/Users/dmitryduev/_jive/pypride/src/pypride/inp.cfg'

    ''' target list [no limits on Vmag] '''
    # date in UTC!!! (for KP, it's the next day if it's still daytime)
    now = datetime.datetime.now(pytz.timezone("America/Phoenix"))
    today = datetime.datetime(now.year, now.month, now.day)

    tl = TargetListPlanetsAndMoons(f_inp, _observatory='kitt peak', _m_lim=16)
    # tl.target_list_all(today)
    targets = tl.target_list_observable(tl.target_list_all(today), today)

    ''' make/change XML files '''
    path = '/Users/dmitryduev/web/qserv/operation'
    program_number = 24
    txml = TargetXML(path=path, program_number=program_number, server='http://localhost:8080')
    # dump 'em targets!
    txml.dumpTargets(targets, epoch='J2000')
