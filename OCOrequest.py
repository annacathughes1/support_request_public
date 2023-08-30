# -*- coding: utf-8 -*-
"""
@author: Anna Hughes
tandem.py adapted from Mr. Paul Sanchez

Discrete simulation model of OCO request process from BDE S6 & CEMA staff
Reference FM 3-12, p125

Version 2: 26 FEB 23
    added revision requirement of above proficiency requirement and ctRevise < 3
    added reviseCheck()
    added cancelCheck()   
    
Version 3: 14 MAR 23
    added output statements
    strategic echelon service times are 1/numpy.random.normal(30,7).
        - "average" time that the request spends above tactical unit level
          is 6 months. there are six queues at strategic level, so "average" 
          time spent at each is 30 days, with standard deviation of 7 days
    tactical echelon service times are 1/numpy.random.normal(7,3).
    requester service times are 1/numpy.random.normal(7,3).
    maxServers (personnel assigned in a team) are numpy.random.randint(1,5).
    availability = numpy.random.random()
    proficiency = numpy.random.random()
    
Version 4: 29 MAR 23
    added output statements for starting proficiency, learning rate, retain rate
    added learnCurve, learnRate
    profCurve = 100 [x,y] values on a random learning curve
                sigmoid CDF of logistic distribution
                range of 100 random values best to make different curves that
                    are still relatively smooth
    yaml functionality

Version 5: 11 APR 23
    added output arrays to request for each state variable
    additional output
    improved yaml functionality
    fixed bug in cancelCheck()   
    added 30 day buffer on to 180 days expected of tMax to increase variability in request status
    
Version 6: 13 MAY
    adjust YAML file & script to run stacked NOLHs with 5 factors, 27 design points
        for better interpretability of results
        
Version 7: 04 JUN 23
    update output per design, per replication;
    fixed errors
          
"""

from simplekit import SimpleKit
import numpy
import math
import queue
import csv
import yaml

DEBUG = False              # Toggle to get data dump for all events
PRODUCTION_RUN = True      # Toggle to use random seeding


##### FUNCTIONS #####


def loadConfig():
    ''' loads yaml file for experiments'''
    with open('config.yaml', 'r') as file: 
        return yaml.safe_load(file)


# adapted from 
# https://stackoverflow.com/questions/24788200/calculate-the-cumulative-distribution-function-cdf-in-python
def getDiscreteCdf(values):
    '''returns the cdf of a distribution'''
    values = (values - numpy.min(values)) / (numpy.max(values) - numpy.min(values))    
    values_sort = numpy.sort(values)
    values_sum = numpy.sum(values)
    values_sums = []
    cur_sum = 0
    for it in values_sort:
        cur_sum += it
        values_sums.append(cur_sum)
    cdf = [values_sums[numpy.searchsorted(values_sort, it)]/values_sum for it in values]
    return cdf


# adapted from 
# https://stackoverflow.com/questions/24788200/calculate-the-cumulative-distribution-function-cdf-in-python
def getLearnCurve():
    '''returns 100 x and y values of a randomly simulated sigmoid curve'''
    rand_values = [numpy.random.logistic() for _ in range(100)]
    cdf = getDiscreteCdf(rand_values)
    x_p = list(zip(rand_values, cdf))
    x_p.sort(key=lambda it: it[0])
    x = [it[0] for it in x_p]
    y = [it[1] for it in x_p]
    return [x,y]


def getLearnRate(list):
    ''' returns the difference in the y-values of the first and last tuple in 
    the list divided by the difference in the x-values of the first and last 
    tuple in the list. Ex: y1-y0 / x1-x0
    if len(list) has one element, no learning has occurred and rate = 0'''
    if list == None:
        return None
    elif type(list) == int:
        return 1
    elif len(list) == 1:
        return 0
    else:
        if (list[0][0] - list[-1][-2]) == 0:
            return None
        else:
            return (list[0][1] - list[-1][-1]) / (list[0][0] - list[-1][-2])
        
        
def getSvcProcessTime(svcDist, svcTimeMean, option):
    '''returns correct mean arrival rate formula to use based on svcDist[0] string
    or returns delay based on distribution ??? could get rid of option'''
    # shifted exponential distribution
    if svcDist == None:
        return 0
    if svcTimeMean == None:
        return 0
    if svcDist == "sexp":
        if option == "mean":
            return svcTimeMean
        else:
            return abs(svcTimeMean*(1-(1/math.sqrt(3))) + numpy.random.exponential(svcTimeMean/math.sqrt(3)))
    # uniform distribution
    if svcDist == "unif":
        if option == "mean":
            return svcTimeMean
        else:
            return numpy.random.uniform(0, 2*svcTimeMean)
    # triangular distribution
    if svcDist == "tri":
        tri = 6*svcTimeMean / (9 + math.sqrt(45))
        if option == "mean":
            return svcTimeMean
        else:
            return numpy.random.triangular(0, tri, 3*svcTimeMean - tri)
    # exponential distribution
    if svcDist == "exp":
        if option == "mean":
            return svcTimeMean
        else:
            return numpy.random.exponential(svcTimeMean)
    

def showStartProficiency(list):
    ''' returns the first proficiency (y-value) in the list
    if len(list) has one (x,y) pair, returns only y-value'''
    if list == None:
        return None
    elif list == 1:
        return 1
    else:
        return list[0][1]
    
    
def setupQ():
    '''returns queue variables to pass through tandem queue format'''
    return numpy.random.randint(1,99), getLearnCurve()


def setupOutput():
    ''' returns dictonary for output state variables'''
    outputDict = {}
    outputDict["BdeS6Cema_Receive"] = 0   
    outputDict["BdeS3_Receive"] = 0
    outputDict["DivG3_Receive"] = 0
    outputDict["CorpsG3_Receive"] = 0
    outputDict["JTFHQ_Receive"] = 0
    outputDict["COIPE_Receive"] = 0
    outputDict["JFHQC_Receive"] = 0
    outputDict["USCYBERCOM_Receive"] = 0
    outputDict["ARCYBER_Approved"] = 0
    outputDict["JTFtask_Approved"] = 0
    outputDict["CorpsG3_Approved"] = 0
    outputDict["DivG3_Approved"] = 0
    outputDict["BdeS3_Approved"] = 0
    outputDict["BdeS6Cema_Approved"] = 0
    return outputDict



##### CLASSES #####


class Request:
    ctRequest = 0
    ctRevise = 0
    # tMax = 180   # average 6 months for the request process after leaving Corps level

    def __init__(self, arrivalTime, tMax):
        Request.ctRequest += 1
        self.ctRequest = Request.ctRequest
        self.ctRevise = Request.ctRevise
        self.arrivalTime = arrivalTime
        # self.tMax = Request.tMax
        self.tMax = tMax
        self.cxlNode = None
        self.bypass = False
        self.arrRateIn = 1
        self.svcDistIn = None
        # dictonaries to store output per queue of each request
        self.profIn = setupOutput()        # all outputs in format:
        self.availIn = setupOutput()       # ["queue.name", output]
        self.meanSvcIn = setupOutput()
        self.retainRateIn = setupOutput()
        self.ctReviseOut = setupOutput()
        self.ctCancelOut = setupOutput()
        self.ctPCSOut = setupOutput()
        self.learnRateOut = setupOutput()

    def timeInSystem(self, currentTime):
        return currentTime - self.arrivalTime
    


class NamedQueue:
    ctCancel = 0   # number of cancelled requests in a queue
    ctPCS = 0      # number of times a trained individual leaves the unit, 
                   ## causing overall proficiency to decrease
    
    # def __init__(self, maxServers = 2, svcDist = None, arrRate = 1, svcTimeMean = 0,
    #              next = None, prev = None, proficiency = 1, learnCurve = None,
    #              learnRate = 1, learnIndex = None, availability = 1, retainRate = 1,
    #              name = "unknown"):
    def __init__(self, maxServers = None, svcDist = None, arrRate = None, svcTimeMean = None,
                 next = None, prev = None, proficiency = None, learnCurve = None,
                 learnRate = None, learnIndex = None, availability = 1, retainRate = 1,
                 name = None, incrementMin = 0, incrementRange = 20):
        self.arrRate = arrRate
        self.meanInterArrivalTime = 1.0 / float(arrRate)
        self.avail = availability
        self.svcDist = svcDist
        self.svcTimeMean = svcTimeMean
        self.numAvailableServers = maxServers
        self.maxServers = maxServers
        self.queue = queue.Queue()
        self.next = next
        self.proficiency = proficiency
        self.learnCurve = learnCurve
        self.learnRate = learnRate
        self.learnIndex = learnIndex
        self.name = name
        self.ctCancel = NamedQueue.ctCancel
        self.ctPCS = NamedQueue.ctPCS
        self.retainRate = retainRate
        self.incrementMin = incrementMin
        self.incrementRange = incrementRange


    def push(self, request):
        self.queue.put(request)


    def pop(self):
        if self.queue.empty():
            return None
        else:
            return self.queue.get()



class Tandem(SimpleKit):
    """Implementation of a tandem queueing model using SimpleKit."""


    def __init__(self, routing, startQueue = None, shutdownTime = 10.0, tMax = 10.0):
        """Construct an instance of the tandem queueing system."""
        SimpleKit.__init__(self)
        self.routing = routing
        self.startQueue = startQueue
        self.shutdownTime = shutdownTime
        self.tMax = tMax


    def init(self):
        """Initialize all state variables, schedule first arrival and the halt."""
        """Note that shutdown stops new arrivals and, but continues in-progress work."""
        self.schedule(self.arrival, 0.0)
        self.schedule(self.shutdown, self.shutdownTime, priority = 0)
        if DEBUG:
            self.dumpState("Init")
        for queuename in ["BdeS6Cema_Receive","BdeS3_Receive","DivG3_Receive","CorpsG3_Receive","JTFHQ_Receive","COIPE_Receive"]:
            queue = self.routing[queuename]
            self.schedule(self.PCS, numpy.random.exponential(queue.retainRate * 365.0 / queue.maxServers), queue)


    def arrival(self):
        """Schedule join of the first queue and the next arrival."""
        queue = self.routing[self.startQueue]
        self.schedule(self.join, 0, queue, Request(self.model_time, self.tMax))
        mean = queue.meanInterArrivalTime
        self.schedule(self.arrival, numpy.random.exponential(mean))
        if DEBUG:
            self.dumpState("Arrival")


    def PCS(self, queue):
        # if the queue is one where proficiency may cause request revisions,
        # and the probability of an individual leaving is high,
        # then the proficiency is reduced by multiplying the current proficiency
        # and the ratio of trained to untrained personnel
        # increment ctPCS
        if queue.name == "BdeS6Cema_Receive" or queue.name == "BdeS3_Receive" or queue.name == "DivG3_Receive" or queue.name == "CorpsG3_Receive" or queue.name == "JTFHQ_Receive" or queue.name == "COIPE_Receive":
            queue.ctPCS += 1
            queue.proficiency = queue.proficiency * (queue.maxServers - 1)/queue.maxServers
            if self.model_time < self.shutdownTime + 2 * self.tMax:
                self.schedule(self.PCS, numpy.random.exponential(queue.retainRate * 365.0 / queue.maxServers), queue)
        if DEBUG:
            self.dumpState("PCS")


    def join(self, queue, request):
        """Enqueue the request, schedule begin service if possible."""
        queue.push(request)
        if queue.numAvailableServers is not None and queue.numAvailableServers > 0:
            self.schedule(self.beginService, 0.0, queue, priority = 2)
        if DEBUG:
            self.dumpState("Join")
            
    
    def cancelCheck(self, queue, request):
        """returns True if a request moved to a cancel node, else returns False"""
        if DEBUG:
            self.dumpState("cancelRequest")
        # if any tactical echelon receives request approval after the alloted time allowed, 
        # then the unit will decide to cancel the support
        if queue.name == "CorpsG3_Approved" or queue.name == "DivG3_Approved" or queue.name == "BdeS3_Approved" or queue.name == "BdeS6Cema_Approved":
            if request.timeInSystem(self.model_time) > request.tMax:
                return True
            else:
                return False


    def reviseCheck(self, queue, request):
        """returns True if a request needs to be revised, else returns False"""
        # if one of the tactical units receives a request where the previous unit's
        # proficiency is not high enough,
        # then the request is returned for revisions and the lower echelon's
        # proficiency increases slightly
        
        if queue.name == "BdeS6Cema_Receive" or queue.name == "BdeS3_Receive" or queue.name == "DivG3_Receive":
            if queue.proficiency < 0.6 and request.ctRevise < 3:        # untrained or less than max revisions
                return True
        # if any request passed to strategic level is given by proficient/untrained team,
        # then mark it for revision
        elif queue.name == "CorpsG3_Receive" or queue.name == "JTFHQ_Receive":
            if queue.proficiency < 0.75 and request.ctRevise < 3:     # proficient or less than max revisions
                return True
        else:
            return False
        
        
    def bypassCheck(self, queue, request):
        '''returns True if a request can bypass to USCYBERCOM'''
        if queue.name == "COIPE_Receive" and queue.proficiency >= 0.9:
            request.bypass = True
            queue.next = "USCYBERCOM_Receive"
    
    
    def updateOutput(self, queue, request):
        '''updates all the queue output for a request
        if statements error-check input from default queue'''
        if queue.learnRate == None:
            request.profIn[queue.name] = 0
            request.learnRateOut[queue.name] = 0
        else:
            request.profIn[queue.name] = showStartProficiency(queue.learnRate)
            request.learnRateOut[queue.name] = getLearnRate(queue.learnRate)
        if queue.svcTimeMean == None:
            request.meanSvcIn[queue.name] = 0
        else:
            request.meanSvcIn[queue.name] = queue.svcTimeMean
        
        request.availIn[queue.name] = queue.avail  
        request.retainRateIn[queue.name] = queue.retainRate
        request.ctReviseOut[queue.name] = request.ctRevise
        request.ctCancelOut[queue.name] = queue.ctCancel
        request.ctPCSOut[queue.name] = queue.ctPCS
        
    
    
    def beginService(self, queue):
        """Remove request from the queue, allocate server, schedule endService.
        Check to see if JFHQC_Receive can be bypassed.
        Check to see if request needs to be revised.
        Check to see if request needs to be cancelled."""
        request = queue.pop()
        
        queue.numAvailableServers -= 1
        delay = abs(getSvcProcessTime(queue.svcDist, queue.svcTimeMean, "delay"))
        if queue.avail is not None:
            delay = delay + (1 - queue.avail)
        
        # if the node is already marked to revise, 
        # then change queue.next to itself to send during endService,
        # increment ctRevise, add new learnRate value, and increase proficiency
        # along learnCurve
        if self.reviseCheck(queue, request) == True:
            request.ctRevise += 1
            if queue.incrementRange == 0:
                queue.incrementRange += 1
            i = numpy.random.randint(queue.incrementMin, queue.incrementMin + queue.incrementRange)   # random increment for learnCurve
            
            # if random increment does not max out proficiency curve
            if queue.learnIndex + i < 99:
                queue.learnIndex = queue.learnIndex + i
            # if random increment does max out proficiency curve
            if queue.learnIndex + i >= 99:
                queue.learnIndex = 99
            
            # update learnRate array with new proficiency
            newProf = (queue.learnCurve[0][queue.learnIndex], queue.learnCurve[1][queue.learnIndex])
            queue.learnRate.append(newProf) 
            
            if queue.proficiency < 1:  
                queue.proficiency = newProf[1]
            
            queue.next = queue.name 
        # else, queue.needsRevision == False and we ensure to reset each queue.next
        # with its original value
        else:
            if queue.name == "BdeS6Cema_Receive":
                queue.next = "BdeS3_Receive"
            elif queue.name == "BdeS3_Receive":
                queue.next = "DivG3_Receive"
            elif queue.name == "DivG3_Receive":
                queue.next = "CorpsG3_Receive"
            elif queue.name == "CorpsG3_Receive":
                queue.next = "JTFHQ_Receive"
            elif queue.name == "JTFHQ_Receive":
                queue.next = "COIPE_Receive"

                
        # if a request needs to be cancelled, 
        # then increment ctCancel, record the queue where it was cancelled,
        # and queue.next set as the appropriate cancel queue
            if self.cancelCheck(queue, request) == True:
                queue.ctCancel += 1
                request.cxlNode = queue.name
                if queue.name == "CorpsG3_Approved":
                    queue.next = "CorpsG3_Cancelled"
                elif queue.name == "DivG3_Approved":
                    queue.next = "DivG3_Cancelled"
                elif queue.name == "BdeS3_Approved":
                    queue.next = "BdeS3_Cancelled"
                elif queue.name == "BdeS6Cema_Approved":
                    queue.next = "BdeS6Cema_Cancelled"
            else:
                if queue.name == "CorpsG3_Approved":
                    queue.next = "DivG3_Approved"
                elif queue.name == "DivG3_Approved":
                    queue.next = "BdeS3_Approved"
                elif queue.name == "BdeS3_Approved":
                    queue.next = "BdeS6Cema_Approved"
                elif queue.name == "BdeS6Cema_Approved":
                    queue.next = None
                
        
        # explicit statement for JTFHQ proficiency is "trained"
        if queue.name == "JTFHQ_Receive" and queue.proficiency >= 0.75:
            queue.next = "COIPE_Receive"


        # if CO-IPE proficiency is "trained", then situational awareness given
        # to USCYBERCOM will allow request to bypass JFHQC_Receive queue
        self.bypassCheck(queue, request)
        
        self.schedule(self.endService, delay, queue, request)
        
        if DEBUG:
            self.dumpState("beginService")
    

    def endService(self, queue, request):
        """Free server, if requests are waiting initiate another service. Pass"""
        """the request to the next queue if one exists."""
        queue.numAvailableServers += 1
        
        # saves request initial arrival Rate and service distribution
        if queue.name == "BdeS6Cema_Receive":
            request.arrRateIn = queue.meanInterArrivalTime
            request.svcDistIn = queue.svcDist
        
        # tMax becomes 6 months plus the time that the request spent in the
        # tactical echelons
        if queue.name == "CorpsG3_Receive":
            request.tMax += request.timeInSystem(self.model_time)
        
        # update request outputs for current queue
        self.updateOutput(queue, request)
        
        if queue.queue.qsize() > 0:
            self.schedule(self.beginService, 0.0, queue, priority = 1)
        if not (queue.next == None):
            self.schedule(self.join, 0.0, self.routing[queue.next], request)

        if DEBUG:
            self.dumpState("endService")        
            print(
           "End service at %-12s" % queue.name,
           "Time: %6.2f" % self.model_time,
           "Request id: %d" % request.ctRequest,
           )
        
        if queue.next == None:
            for key, value in request.learnRateOut.items():
                if value is None:
                    request.learnRateOut[key] = 1 
            row = [# MOE
                   request.timeInSystem(self.model_time),
                   # factors
                   series + 1,
                   num + 1,
                   queue.arrRate,
                   queue.svcDist,
                   queue.svcTimeMean,
                   queue.maxServers,
                   df[series]["availability"],
                   df[series]["retainRate"],
                   df[series]["incrementMin"],
                   df[series]["incrementRange"],
                   # additional output
                   self.model_time,
                   request.ctRequest,
                   request.tMax, 
                   request.cxlNode,
                   request.bypass,
                   # average statistics of request
                   sum(request.profIn.values()) / len(request.profIn.values()),
                   sum(request.ctReviseOut.values()) / len(request.ctReviseOut.values()),
                   sum(request.ctPCSOut.values()) / len(request.ctPCSOut.values()),
                   sum(request.learnRateOut.values()) / len(request.learnRateOut.values()),

                   # detailed output per queue
                   request.profIn["BdeS6Cema_Receive"], 
                   request.profIn["BdeS3_Receive"],
                   request.profIn["DivG3_Receive"],
                   request.profIn["CorpsG3_Receive"],
                   request.profIn["JTFHQ_Receive"],
                   request.profIn["COIPE_Receive"],
                    
                   request.ctReviseOut["BdeS6Cema_Receive"], 
                   request.ctReviseOut["BdeS3_Receive"],
                   request.ctReviseOut["DivG3_Receive"],
                   request.ctReviseOut["CorpsG3_Receive"],
                   request.ctReviseOut["JTFHQ_Receive"],
                   request.ctReviseOut["COIPE_Receive"],

                   request.ctPCSOut["BdeS6Cema_Receive"], 
                   request.ctPCSOut["BdeS3_Receive"],
                   request.ctPCSOut["DivG3_Receive"],
                   request.ctPCSOut["CorpsG3_Receive"],
                   request.ctPCSOut["JTFHQ_Receive"],
                   request.ctPCSOut["COIPE_Receive"],
                   
                   request.learnRateOut["BdeS6Cema_Receive"], 
                   request.learnRateOut["BdeS3_Receive"],
                   request.learnRateOut["DivG3_Receive"],
                   request.learnRateOut["CorpsG3_Receive"],
                   request.learnRateOut["JTFHQ_Receive"],
                   request.learnRateOut["COIPE_Receive"]]
            with open('output.csv', 'a', newline='') as out:
                writer = csv.writer(out, delimiter=',')
                writer.writerow(row)
            

    def shutdown(self):
        """Close shop by shutting doors, i.e., no more arrivals. Finish"""
        """serving existing requests"""
        self.cancel_next(self.arrival)
        self.cancel_all(self.PCS)
        if DEBUG:
            self.dumpState("shutdown")
            

    def dumpState(self, event):
        """Dump of the current state of the model."""
        print("Time: %6.2f" % self.model_time, "  Event: %-12s" % event)
        for name in self.routing:
            q = self.routing[name]
            print(name,  ":  Queue Length: %3d" % q.queue.qsize(),
                " Available Servers: ", q.numAvailableServers)


##### END SETUP ##### 


if __name__ == '__main__':
    if not PRODUCTION_RUN:
        numpy.random.seed(12345)    # For reproducibility.
    # Instantiate and run a copy of the tandem model.
        
    # heading for output.csv
    heading = [ # MOE
                "requestTIS", 
                # factors
                "DP",
                "run",
                "arrivalRate",
                "svcDist",
                "svcTimeMean",
                "maxServers",
                "availability",
                "retainRate",
                "incrementMin",
                "incrementRange",
                # additional output
                "modelTime", 
                "requestID", 
                "requestMaxTimeAllowed", 
                "cancelledBy",
                "requestBypass",
                #average statistics of request
                "avgProficiencyIN",
                "avgReviseOUT",
                "avgPCSOUT",
                "avgLearnRateOut",
                
                # detailed output per queue
                "request.profIn_BdeS6Cema_Receive", 
                "request.profIn_BdeS3_Receive",
                "request.profIn_DivG3_Receive",
                "request.profIn_CorpsG3_Receive",
                "request.profIn_JTFHQ_Receive",
                "request.profIn_COIPE_Receive",

                "request.ctReviseOut_BdeS6Cema_Receive", 
                "request.ctReviseOut_BdeS3_Receive",
                "request.ctReviseOut_DivG3_Receive",
                "request.ctReviseOut_CorpsG3_Receive",
                "request.ctReviseOut_JTFHQ_Receive",
                "request.ctReviseOut_COIPE_Receive",

                "request.ctPCSOut_BdeS6Cema_Receive", 
                "request.ctPCSOut_BdeS3_Receive",
                "request.ctPCSOut_DivG3_Receive",
                "request.ctPCSOut_CorpsG3_Receive",
                "request.ctPCSOut_JTFHQ_Receive",
                "request.ctPCSOut_COIPE_Receive",
                   
                "request.learnRateOut_BdeS6Cema_Receive", 
                "request.learnRateOut_BdeS3_Receive",
                "request.learnRateOut_DivG3_Receive",
                "request.learnRateOut_CorpsG3_Receive",
                "request.learnRateOut_JTFHQ_Receive",
                "request.learnRateOut_COIPE_Receive"
                ]    
    with open('output.csv', 'w', newline='') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(heading)
        
    df = loadConfig()
    
    # number of runs for each yaml series   
    runs = 50
    
    for series in range(0, len(df)):
    
        for num in range(0, runs):
            # identify starting proficieny curve index for each unit that requires it
            i_BdeS6Cema, curve_BdeS6Cema = setupQ()
            i_BdeS3, curve_BdeS3 = setupQ()
            i_DivG3, curve_DivG3 = setupQ()
            i_CorpsG3, curve_CorpsG3 = setupQ()
            i_JTFHQ, curve_JTFHQ = setupQ()
            i_COIPE, curve_COIPE = setupQ()
        
            routing = {
                "BdeS6Cema_Receive":  NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],   next = "BdeS3_Receive",      availability = df[series]["availability"], proficiency = curve_BdeS6Cema[1][i_BdeS6Cema], learnCurve = curve_BdeS6Cema, learnRate = [(curve_BdeS6Cema[0][i_BdeS6Cema], curve_BdeS6Cema[1][i_BdeS6Cema])], learnIndex = i_BdeS6Cema, retainRate = df[series]["retainRate"], name = "BdeS6Cema_Receive", arrRate = df[series]["arrivalRate"], incrementMin = df[series]["incrementMin"], incrementRange = df[series]["incrementRange"]),
                "BdeS3_Receive":      NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],   next = "DivG3_Receive",      availability = df[series]["availability"], proficiency = curve_BdeS3[1][i_BdeS3],         learnCurve = curve_BdeS3,     learnRate = [(curve_BdeS3[0][i_BdeS3],         curve_BdeS3[1][i_BdeS3])],         learnIndex = i_BdeS3,     retainRate = df[series]["retainRate"], name = "BdeS3_Receive", arrRate = df[series]["arrivalRate"], incrementMin = df[series]["incrementMin"], incrementRange = df[series]["incrementRange"]),
                "DivG3_Receive":      NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],   next = "CorpsG3_Receive",    availability = df[series]["availability"], proficiency = curve_DivG3[1][i_DivG3],         learnCurve = curve_DivG3,     learnRate = [(curve_DivG3[0][i_DivG3],         curve_DivG3[1][i_DivG3])],         learnIndex = i_DivG3,     retainRate = df[series]["retainRate"], name = "DivG3_Receive", arrRate = df[series]["arrivalRate"], incrementMin = df[series]["incrementMin"], incrementRange = df[series]["incrementRange"]),
                "CorpsG3_Receive":    NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],   next = "JTFHQ_Receive",      availability = df[series]["availability"], proficiency = curve_CorpsG3[1][i_CorpsG3],     learnCurve = curve_CorpsG3,   learnRate = [(curve_CorpsG3[0][i_CorpsG3],     curve_CorpsG3[1][i_CorpsG3])],     learnIndex = i_CorpsG3,   retainRate = df[series]["retainRate"], name = "CorpsG3_Receive", arrRate = df[series]["arrivalRate"], incrementMin = df[series]["incrementMin"], incrementRange = df[series]["incrementRange"]),
                "JTFHQ_Receive":      NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],  next = "COIPE_Receive",      availability = df[series]["availability"], proficiency = curve_JTFHQ[1][i_JTFHQ],         learnCurve = curve_JTFHQ,     learnRate = [(curve_JTFHQ[0][i_JTFHQ],         curve_JTFHQ[1][i_JTFHQ])],         learnIndex = i_JTFHQ,    retainRate = df[series]["retainRate"], name = "JTFHQ_Receive", arrRate = df[series]["arrivalRate"], incrementMin = df[series]["incrementMin"], incrementRange = df[series]["incrementRange"]),
            
                "COIPE_Receive":      NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],  next = "JFHQC_Receive",      availability = df[series]["availability"], proficiency = curve_COIPE[1][i_COIPE],         learnCurve = curve_COIPE,     learnRate = [(curve_COIPE[0][i_COIPE],         curve_COIPE[1][i_COIPE])],         learnIndex = i_COIPE,    retainRate = df[series]["retainRate"], name = "COIPE_Receive", arrRate = df[series]["arrivalRate"], incrementMin = df[series]["incrementMin"], incrementRange = df[series]["incrementRange"]),
                "JFHQC_Receive":      NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],  next = "USCYBERCOM_Receive", availability = df[series]["availability"], name = "JFHQC_Receive", arrRate = df[series]["arrivalRate"]),
                "USCYBERCOM_Receive": NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],  next = "ARCYBER_Approved",   availability = df[series]["availability"], name = "USCYBERCOM_Receive", arrRate = df[series]["arrivalRate"]),
            
                "ARCYBER_Approved":   NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],  next = "JTFtask_Approved",   availability = df[series]["availability"], name = "ARCYBER_Approved", arrRate = df[series]["arrivalRate"]),
                "JTFtask_Approved":   NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],  next = "CorpsG3_Approved",   availability = df[series]["availability"], name = "JTFtask_Approved", arrRate = df[series]["arrivalRate"]),
                
                "CorpsG3_Approved":   NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],   next = "DivG3_Approved",     availability = df[series]["availability"], name = "CorpsG3_Approved", arrRate = df[series]["arrivalRate"]),
                "CorpsG3_Cancelled":  NamedQueue(name = "CorpsG3_Cancelled", arrRate = df[series]["arrivalRate"], svcDist = None, svcTimeMean = 0, availability = 1),
                "DivG3_Approved":     NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],   next = "BdeS3_Approved",     availability = df[series]["availability"], name = "DivG3_Approved", arrRate = df[series]["arrivalRate"]),
                "DivG3_Cancelled":    NamedQueue(name = "DivG3_Cancelled", arrRate = df[series]["arrivalRate"], svcDist = None, svcTimeMean = 0, availability = 1),
                "BdeS3_Approved":     NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],   next = "BdeS6Cema_Approved", availability = df[series]["availability"], name = "BdeS3_Approved", arrRate = df[series]["arrivalRate"]),
                "BdeS3_Cancelled":    NamedQueue(name = "BdeS3_Cancelled", arrRate = df[series]["arrivalRate"], svcDist = None, svcTimeMean = 0,  availability = 1),
                "BdeS6Cema_Approved": NamedQueue(maxServers = df[series]["maxServers"], svcTimeMean = df[series]["meanSvc"], svcDist = df[series]["svcDist"],   name = "BdeS6Cema_Approved", arrRate = df[series]["arrivalRate"]),
                "BdeS6Cema_Cancelled":NamedQueue(name = "BdeS6Cema_Cancelled", arrRate = df[series]["arrivalRate"], svcDist = None, svcTimeMean = 0, availability = 1)
                }
            model = Tandem(routing, startQueue = "BdeS6Cema_Receive", shutdownTime = 365.0, tMax = 180).run()
            print("Series", series + 1,"with run", num + 1,"complete")
    
    print("All series and runs complete.")

