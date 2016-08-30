# -*- coding: utf-8 -*-
"""
Created on Wed Sep 02 17:34:44 2015

@author: jbooth
"""

myD = {'name': 'Juliann', 'age':21}
myD['height'] = 65

for theName in myD:
    print(theName + '\t')
    myList = myD[theName]
    for a in myList:
        print(a, end= ' ')
    print('\n')

import random
position = 0
walk = [position] # Start at the origin
steps = 100
for i in range(steps):
    if random.randint(0,1):
        step = 1
    else:
        step = -1
    position += step
    walk.append(position)
