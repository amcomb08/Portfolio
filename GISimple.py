import random

G = 0
I = 5

impSpd = tuple([1,9])
golemSpd = tuple([3,5])
headStart = 5
exitPosition = 50

def GISimple(impSpd, golemSpd, headStart, exitPosition):

    #intitalizing the position of Golem imp and time
    golem = 0
    imp = headStart
    time=0
    impMax = impSpd[1]
    impMin = impSpd[0]
    golemMax = golemSpd[1]
    golemMin = golemSpd[0]
    while(True):
        golem = golem + random.randrange(golemMin,golemMax)
        imp = imp  + random.randrange(impMin,impMax)
        if imp>=exitPosition :
            return True
        if golem>=imp :
            return False