# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:31:43 2013

@author: wthelen
Updated: 2022 July 7 by Jay Wellik
"""
import sys
import time
import datetime
#import pdb
#import timezones
#import dateutil.parser
#import datetime
import numpy as np

def parseData(sumline, startind, totalLength, numDec):
    nowData = None
    blankStr=' ' * (totalLength)
    data = sumline[startind:startind+totalLength]
#    print blankStr
#    print data
    if data != blankStr and data:
        if numDec == 0:
            nowData = int(data)
        elif numDec == 1:
            nowData = float(data)/10
        elif numDec == 2:
            nowData = float(data)/100
        elif numDec == 3:
            nowData = float(data)/1000
        elif numDec == 4:
            nowData = float(data)/10000
    return nowData

class hypoSta:
    def __init__(self):
        self.station = ''
        self.network = ''
        self.compCode = ''
        self.channel = ''
        self.pRemark = ''
        self.pFM = ''
        self.pWeight = []
        self.pArrival = []
        self.pRes = []
        self.pWeightTrue = []
        self.sArrival = []
        self.sRemark = ''
        self.sWeight = []
        self.sRes = []
        self.amp = []
        self.ampUnits = ''
        self.sWeightTrue = []
        self.pDelay = []
        self.sDelay = []
        self.delta = []
        self.emergenceAngle = []
        self.ampMagWeight = ''
        self.durMagWeight = ''
        self.ampPeriod = []
        self.staRemark = []
        self.codaDur = []
        self.azimuth = []
        self.durMag = []
        self.ampMag = []
        self.pImport = []
        self.sImport = []
        self.dataSource = ''
        self.durMagType = ''
        self.ampMagType = ''
        self.location = ''
        self.ampType = ''
        self.altStaCode = ''
        self.ampMagUsed = ''
        self.durMagUsed = ''
        return
        
    def parseStaLine(self, staline):
        sta = staline[0:5]
        self.station=sta.rstrip()
        self.network = staline[5:7]
        self.compCode = staline[8:9]
        self.channel = staline[9:12]
        pSec = staline[29:34]         # Checking for existance of p-wave
        if len(pSec.rstrip())!= 0:
            self.pRemark=staline[13:15]
            self.pFM=staline[15:16]
            self.pWeight=parseData(staline, 16, 1 ,0)
            sec = parseData(staline, 29, 3, 0)
            if sec == None:
                sec = 0
            msec = parseData(staline, 32, 2, 0)*10000
            if msec == None:
                msec = 0
            self.pArrival = datetime.datetime(int(staline[17:21]), int(staline[21:23]), int(staline[23:25]), int(staline[25:27]), int(staline[27:29]), sec, msec)
            self.pRes = parseData(staline, 34, 4, 2)
            self.pWeightTrue = parseData(staline, 38, 3, 2)
            self.pDelay = parseData(staline, 66, 4, 2)
            self.pImport = parseData(staline,100, 4, 3)
        sSec = staline[41:46]       # Checking for existance of s-wave
        if len(sSec.rstrip())!= 0:
            self.sRemark=staline[46:48]
            self.sWeight=parseData(staline, 49, 1 ,0)
            sec = parseData(staline, 41, 3, 0)
            if sec == None:
                sec = 0
            msec = parseData(staline, 44, 2, 0)*10000
            if msec == None:
                msec = 0
            plusSec = None
            if sec > 59:   # This is a case where the sec > 60
                plusSec = datetime.timedelta(minutes=1)
                sec =sec-60
            self.sArrival = datetime.datetime(int(staline[17:21]), int(staline[21:23]), 
                                              int(staline[23:25]), int(staline[25:27]), 
                                            int(staline[27:29]), sec , msec)
            if plusSec:
                self.sArrival = self.sArrival + plusSec
            self.sRes = parseData(staline, 50, 4, 2)
            self.sWeightTrue = parseData(staline, 63, 3, 2)
            self.sDelay = parseData(staline, 70, 4, 2)
            self.sImport = parseData(staline,104, 4, 3)
        self.amp = parseData(staline, 54, 7, 2)
        self.ampType = parseData(staline, 61, 2, 0)
        self.delta = parseData(staline, 74, 4, 1)
        self.emergenceAngle = parseData(staline, 78, 3, 0)
        self.ampMagWeight = parseData(staline, 81, 1, 0)
        self.durMagWeight = parseData(staline, 82, 1, 0)
        self.ampPeriod = parseData(staline, 83, 3, 2)
        self.staRemark = staline[86:87]
        self.codaDur = parseData(staline, 94, 3, 2)
        self.azimuth = parseData(staline, 91, 3, 0)
        self.durMag = parseData(staline, 94, 3, 2)
        self.ampMag = parseData(staline, 97, 3, 2)
        self.dataSource = staline[108:109]
        self.durMagType = staline[109:110]
        self.ampMagType = staline[110:111]
        self.location = staline[111:113]
        return
    '''
    def __str__(self,stream=None):
        stream = stream or sys.stderr            
        stream.write('SEED String: %s.%s.%s.%s\n' % (self.network, self.station, self.location, self.channel))
        stream.write('Component Code: %s\n' % (self.compCode))
        stream.write('Distance to earthquake: %0.1f\n' % (self.delta))
        stream.write('Azimuth to earthquake: %d, Emergence Angle: %d\n' % (self.azimuth, self.emergenceAngle))
        stream.write('Data source: %s\n' % (self.dataSource))
        stream.write('Station Remark: %s\n' % (self.staRemark))
        stream.write('P wave parameters:\n')
        if self.pArrival:
            stream.write('	Pick time = %s\n' % (self.pArrival))
            stream.write('	Pick remark = %s, Pick FM = %s, Pick Weight = %d\n' % (self.pRemark, self.pFM, self.pWeight))
            stream.write('	Pick residual = %0.2f, Pick delay = %0.2f, Pick True Weight = %0.2f\n' % (self.pRes, self.pDelay, self.pWeightTrue))
            stream.write('	Pick importance = %0.3f\n' % (self.pImport))
        else:
            stream.write('	No pick\n')
        stream.write('S wave parameters:\n')
        if self.sArrival:
            stream.write('	Pick time = %s\n' % (self.sArrival))
            stream.write('	Pick remark = %s,  Pick Weight = %d\n' % (self.sRemark, self.sWeight))
            stream.write('	Pick residual = %0.2f, Pick delay = %0.2f, Pick True Weight = %0.2f\n' % (self.sRes, self.sDelay, self.sWeightTrue))
            stream.write('	Pick importance = %0.3f\n' % (self.sImport))
        else:
            stream.write('	No pick\n')
        stream.write('Amplitude parameters:\n')
        if self.amp:
            stream.write('	Amplitude = %0.2f\n' % (self.amp))
            stream.write('	Amplitude Units = %d (0 = PP mm, 1 = Zero to Peak mm, 2 = digital counts)\n' % (self.amp))
            stream.write('	Amplitude Period = %0.2f\n' % (self.ampPeriod))
            stream.write('	Amplitude Mag = %0.2f, Mag Type = %s, Amp Mag Weight = %d\n' % (self.ampMag, self.ampMagType, self.ampMagWeight))
        else:
            stream.write('	No Amplitude Information \n')
        stream.write('Duration parameters:\n')
        if self.codaDur:
            stream.write('	Duration = %0.2f\n' % (self.codaDur))
            stream.write('	Duration Mag = %0.2f, Mag Type = %s, Dur Mag Weight = %d\n' % (self.durMag, self.durMagType, self.durMagWeight))
        else:
            stream.write('	No Duration Information\n')
        return 'Populated HypoSum object\n'
        
        '''
        
class hypoSum:
    def __init__(self, originTime=None,lat=None, lon=None, depth=None, mag=None, magNumPha=None, magAmp=None, numPha=None, gap=None, dmin=None, rms=None, err=None, errAz=None, errDip=None,herr=None, verr=None, magCoda=None, region=None, remark=None, fix=None, magEx=None, dbid=None, velmodel=None, auxRemarkAnalyst=None, auxRemarkHypo=None,locRemark=None, errSmall=None, numPhaS=None, numPol=None, nAmpMag=None, nCodaMag=None,medianAmp=None, medianDur=None, authCode=None, netCodeLoc=None, netCodeDur=None, netCodeAmp=None, durType=None, numPhaTotal=None, ampType=None, magExtType=None, magExt=None, magExtWeight=None, magAmpAltType=None, magAmpAlt=None, magAmpAltWeight=None, magType=None, magCodaAltType=None, magCodaAlt=None, magCodaAltWeight=None, ver=None, verReview=None, domain=None, versionLoc=None ):
        self.originTime = None
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.mag = mag
        self.magNumPha = magNumPha
        self.magAmp = magAmp
        self.numPha = numPha
        self.gap = gap
        self.dmin = dmin
        self.rms = rms
        self.err = err
        self.errAz = errAz
        self.errDip = errDip
        self.errInt = err
        self.errDepInt = errDip
        self.errAzInt = errAz
        self.herr = herr
        self.verr = verr
        self.magCoda = magCoda
        self.locRemark = locRemark
        self.errSmall = errSmall  
        self.auxRemarkAnalyst = auxRemarkAnalyst
        self.auxRemarkHypo = auxRemarkHypo
        self.numPhaS = numPhaS
        self.region = region
        self.remark = remark
        self.fix = fix
        self.numPol = numPol
        self.nAmpMag = nAmpMag
        self.nCodaMag = nCodaMag
        self.medianAmp = medianAmp
        self.medianDur = medianDur
        self.velmodel = velmodel
        self.authCode = authCode
        self.netCodeLoc = netCodeLoc
        self.netCodeDur = netCodeDur
        self.netCodeAmp = netCodeAmp
        self.durType = durType
        self.numPhaTotal = numPhaTotal
        self.ampType = ampType
        self.magExtType = magExtType
        self.magExt = magExt
        self.magExtWeight = magExtWeight
        self.magAmpAltType = magAmpAltType
        self.magAmpAlt = magAmpAlt
        self.magAmpAltWeight = magAmpAltWeight
        self.dbid = dbid
        self.magType = magType
        self.mag = mag
        self.magNumPha = magNumPha
        self.magCodaAltType = magCodaAltType
        self.magCodaAlt = magCodaAlt
        self.magCodaAltWeight = magCodaAltWeight
        self.ver = ver
        self.reviewFlag = verReview
        self.domain = domain
        self.versionLoc = versionLoc
        self.modelType = None
        self.depthType = None
        self.depthDatum = None
        self.geoidDepth = None
        self.picks = {}
        
    '''
    def __str__(self,stream=None):
        stream = stream or sys.stderr            
        stream.write('Origin Time = %s\n' % self.originTime)
        stream.write('Loc Parameters:\n')
        stream.write('Lat: %s ' % self.lat)
        stream.write('Lon: %s ' % self.lon)
        stream.write('Depth: %s ' % self.depth)
        if self.depthType:
            stream.write('Depth Type: %s Model Type: %s Depth Datum: %d Geoid Depth: %f ' % (self.depthType, self.modelType, self.depthDatum, self.geoidDepth))
        stream.write('Mag: %s\n' % self.mag)
        stream.write('Error Stats:\n')
        stream.write('Vert Err: %s ' % self.verr)
        stream.write('Horiz Err: %s ' % self.herr)
        stream.write('RMS: %s ' % self.rms)
        stream.write('Delta: %s ' % self.dmin)
        stream.write('Gap: %s\n' % self.gap)
        stream.write('Station Parameters:\n')
        stream.write('Num Phase: %s ' % self.numPha)
        stream.write('Num S Phase: %s ' % self.numPhaS)
        stream.write('Total Phases: %s\n' % self.numPhaTotal)
        stream.write('Preferred Magnitude Params:\n')
        stream.write('Mag: %s ' % self.mag)
        stream.write('MagType: %s ' % self.magType)
        stream.write('MagNumPha: %s\n' % self.magNumPha)
        stream.write('Coda Magnitude Params:\n')
        stream.write('Mag: %s ' % self.magCoda)
        stream.write('MagNumPha: %s ' % self.nCodaMag)
        stream.write('AltMag: %s ' % self.magCodaAlt)
        stream.write('AltMagType: %s ' % self.magCodaAltType)
        stream.write('AltMagWeight: %s\n' % self.magCodaAltWeight)
        stream.write('Amp Magnitude Params:\n')
        stream.write('Mag: %s ' % self.magAmp)
        stream.write('MagNumPha: %s ' % self.nAmpMag)
        stream.write('AltMag: %s ' % self.magAmpAlt)
        stream.write('AltMagType: %s ' % self.magAmpAltType)
        stream.write('AltMagTypeWeight: %s\n' % self.magAmpAltWeight)
        stream.write('External Magnitude Params:\n')
        stream.write('Mag: %s ' % self.magExt)
        stream.write('MagType: %s ' % self.magExtType)
        stream.write('MagNumPha: %s\n' % self.magExtWeight)
        stream.write('Location params_generic:\n')
        stream.write('Velocity Model: %s ' % self.velmodel)
        stream.write('Region: %s\n' % self.region)
        stream.write('Version Info:\n')
        stream.write('Version: %s ' % self.ver)
        stream.write('ReviewFlag: %s ' % self.reviewFlag)
        stream.write('ID: %s ' % self.dbid)
        stream.write('Domain: %s ' % self.domain)
        stream.write('VersionLoc: %s\n' % self.versionLoc)
        if self.originTime:
            return 'initalized hypoinverse sum object\n'
        else:
            return 'Populated HypoEq object\n'
            
    '''
            
    def parseSumLine(self, sumline):
        self.originTime = datetime.datetime(int(sumline[0:4]), int(sumline[4:6]), int(sumline[6:8]), int(sumline[8:10]), int(sumline[10:12]), int(sumline[12:14]), int(sumline[14:16])*10000)
        latdeg = parseData(sumline, 16, 2, 0)
        latmin = parseData(sumline, 19, 4, 2)/60
        if latmin == None:
            latmin = 0        
        self.lat = latdeg+latmin
        if sumline[18]=='S':
            self.lat = -self.lat
        londeg = parseData(sumline, 23, 3, 0)
        lonmin = parseData(sumline, 27, 4, 2)/60
        if lonmin == None:
            lonmin = 0
        lon = londeg+lonmin
        self.lon = -lon
        if sumline[26]=='E':
            self.lon = -self.lon
        try:
            self.depth = float(sumline[32:36])/100
        except:
            self.depth = np.nan
        self.magAmp = parseData(sumline, 36, 3, 2)
        self.numPha = parseData(sumline, 39, 3, 0)
        self.gap = parseData(sumline, 42, 3, 0)
        self.dmin = parseData(sumline, 45, 3, 0)
        self.rms = parseData(sumline, 48, 4, 2)
        self.err = parseData(sumline, 57, 4, 2)    #Size of largest principal error(km)
        self.errAz = parseData(sumline, 52, 3, 0)        #Azimuth of lgst principal error (deg E of N)
        self.errDip = parseData(sumline, 55, 2, 0)       #Dip of lgst principal error (deg)
        self.errInt = parseData(sumline, 66, 4, 2)     # Intermediate error stats
        self.errAzInt = parseData(sumline, 61, 3, 0)
        self.errDipInt = parseData(sumline, 64, 2, 0)
        self.magCoda = parseData(sumline, 70, 3, 2)   #coda magnitude
        self.locRemark = sumline[73:76];
        self.errSmall = parseData(sumline, 76, 4, 2)  # Smallest error (km)
        self.auxRemarkAnalyst = sumline[80]
        self.fix = sumline[81]
        self.numPhaS = parseData(sumline, 82, 3, 0)          # Number of used s-phases
        self.herr = parseData(sumline, 85, 4, 2)       # Horizontal error
        self.verr = parseData(sumline, 89, 4, 2)       # Vertical error
        self.numPol = parseData(sumline, 93, 3, 0)
        self.nAmpMag = parseData(sumline, 96, 4, 1)
        self.nCodaMag = parseData(sumline, 100, 4, 1)
        self.medianAmp =parseData(sumline, 104, 3, 2)
        self.medianDur =parseData(sumline, 107, 3, 2)
        self.velmodel = sumline[110:113]
        self.authCode = sumline[113]
        self.netCodeLoc = sumline[114]
        self.netCodeDur = sumline[115]
        self.netCodeAmp = sumline[116]
        self.durType = sumline[117]
        self.numPhaTotal = parseData(sumline, 118, 3, 0)   # Number of phases w/ weigtht>0
        self.ampType = sumline[121]
        self.magExtType = sumline[122]
        self.magExt = parseData(sumline, 123, 3, 2)
        self.magExtWeight = parseData(sumline, 126, 3, 1)
        self.magAmpAltType = sumline[129]
        self.magAmpAlt = parseData(sumline, 130, 3, 2)
        self.magAmpAltWeight = parseData(sumline, 133, 3, 1)
        self.dbid = parseData(sumline, 136, 10, 0)
        self.magType = sumline[146]
        self.mag = parseData(sumline, 147, 3, 2)
        self.magNumPha = parseData(sumline, 150, 4, 1)
        self.magCodaAltType = sumline[154]
        self.magCodaAlt =parseData(sumline, 155, 3, 2)
        self.magCodaAltWeight = parseData(sumline, 158, 4, 1)
        self.ver = parseData(sumline, 162, 1, 0)
        if len(sumline) > 164:
            self.reviewFlag = sumline[163]
            self.domain = sumline[164:166]
            self.versionLoc = sumline[166:168]
        if len(sumline) > 169:
            self.depthType = sumline[168]
            self.modelType = sumline[169]
            self.depthDatum = parseData(sumline, 170, 4, 0)
            self.geoidDepth = parseData(sumline, 174, 5, 2)
        
    
    def inRectangle(self, minlat, maxlat, minlon, maxlon):
        if self.lat > minlat and self.lat < maxlat and self.lon > minlon and self.lon < maxlon:
            return 1
        else:
            return 0
    
    #shift time to local
    def getLocalTime(self, offsetHours):
        date = self.originTime
        utcSec = time.mktime(date.timetuple())
        local = time.localtime(utcSec + 3600*offsetHours)
        return time.strftime("%a %b %d, %Y %H:%M:%S %Z", local)
    

    #establish time parameter for event to assign (in javascript) proper color icon.
    def getIcon(self):
        date = self.originTime        
        #utc = time.strptime(date, "%Y, %m, %d, %H, %M, %S")
        #utcSec = time.mktime(utc)
        utcSec = time.mktime(date.timetuple())
        now = time.mktime(time.gmtime())#current utc time
        if( now - utcSec <=7200): #two hours
            return 0
        elif(now - utcSec  <= 172800): #two-weeks
            return 1
        else:
            return 2

class hypoCatalog:
    def __init__(self):
        self.starttime=None
        self.endtime=None
        self.data=[]
        return
    def purgeByTimeWindow(self,start,end):
        '''
        Go through class data and remove all entries not
        in time window.
        '''
        if not self.data: #noop on empty data
            return
        #test for active bounds
        trimstart=self.starttime<start
        trimend=self.endtime>end
        if trimstart: 
            self.starttime=start
        if trimend: 
            self.endtime=end
        if trimstart or trimend:  
            newdat=[]
            for e in self.data:
                add = 1
                if trimstart:
                    add = e.originTime>=start
                if trimend:
                    add = add and e.originTime<=end
                if add:
                    newdat.append(e)
            self.data=newdat
        return
    def purgeByRectangle(self, minlat, maxlat, minlon, maxlon):
        '''
        Purge catalog class for data outside of rectangle
        '''
        if not self.data:
            return
        newdat=[]
        for e in self.data:
            isin = e.inRectangle(minlat, maxlat, minlon, maxlon)
            if isin:
                newdat.append(e)
        self.data=newdat
        return
    def readSumFile(self, filename):
        fid = open( filename, encoding="windows-1254" )
        allLines = fid.readlines()
        fid.close()
        for line in allLines:
            e = hypoSum()
            e.parseSumLine(str(line.rstrip()))
            #pdb.set_trace()
            if self.starttime:
                if e.originTime<self.starttime:
                    self.starttime=e.originTime
            else:
                self.starttime=e.originTime
            if self.endtime:
                if e.originTime>self.endtime:
                    self.endtime=e.originTime
            else:
                self.endtime=e.originTime
            self.data.append(e)
        return   
    def readArcFile(self, filename):
        fid = open( filename, "r" )
        allLines = fid.readlines()
        fid.close()
        e = None
        for line in allLines:
            if line[0] != "$":  # JJW
                if not e:                     #new event
                    if line:                  #not empty
                        e = hypoSum()
                        e.parseSumLine(line.rstrip())
                elif line[0:4].strip()=='':   #end of event marker
                    if self.starttime:
                        if e.originTime<self.starttime:
                            self.starttime=e.originTime
                    else:
                        self.starttime=e.originTime
                    if self.endtime:
                        if e.originTime>self.endtime:\
                            self.endtime=e.originTime
                    else:
                        self.endtime=e.originTime
                    self.data.append(e)             #append to catalog
                    e=None
                else:                         #station line
                    ec = hypoSta()
                    ec.parseStaLine(line.rstrip())
                    stastring = '%s.%s.%s.%s' % (ec.network, ec.station, ec.location, ec.channel)
                    e.picks[stastring]=ec
        return

    def sortByOriginTime(self,reverse=False):
        '''
        sorts data by originTime in increasing order
        optional True reverse argument will reverse sort order
        '''
        self.data.sort(lambda x,y: cmp(x.originTime,y.originTime))
        if reverse:
            self.data.reverse()
        return
    def purge(self):
        self.data=[]
        self.starttime=self.endtime=None
        return
    def writeNEIC(self, filename='testNEIC.csv'):
        '''
        Write NEIC CSV file format
        
        Input
        --------
        filename: output file name
        
        '''
        f = open(filename, 'w')
        for eq in self:
            timestr = eq.originTime.strftime('%Y-%m-%d %H:%M:%S')
            try:
                linestr = '%s, %s, %0.4f, %0.4f, %0.2f, %0.1f, earthquake\n' % (eq.dbid, timestr, eq.lat, eq.lon, eq.depth, eq.mag)
                f.write(linestr)
            except:
                print (eq)
            
        f.close()
        return
    def writeXYZM(self, filename='out.xyzm'):
        '''
        Writes a space-delimted text file with lon(x), lat(y), depth(z), mag(m). 
        Depth is in km.
        
        Input
        ---------
        filename: output file name
        
        '''
        with open(filename, 'w') as f:
            for eq in self:
                f.write('%0.8f %0.8f %0.3f %0.1f\n' % (eq.lon, eq.lat, eq.depth, eq.mag))
                
    def writeObspyCatalog(self):
        '''
        Writes an Obspy Catalog Object from a hypoinverse object
        
        '''
        # print("Buyer beware.  May not be quite right.  Definitely doesn't bring in picks or station mags.")
        from obspy import Catalog, UTCDateTime, __version__
        from obspy.core.event import (Arrival, Comment, CreationInfo, Event, Origin,
                              OriginQuality, OriginUncertainty, Pick,
                              WaveformStreamID, Magnitude, StationMagnitude)
        from obspy.geodetics import (kilometer2degrees, locations2degrees)
        from numpy import median
        cat = Catalog()
        cat.creation_info.creation_time = UTCDateTime()
        cat.creation_info.version = "ObsPy %s" % __version__
        for nowEvent in self:
             oevt = Event()
             o = Origin()
             oevt.origins = [o]
             if not nowEvent.domain:
                 domain = 'us.usgs.vhp'
             o.resource_id = 'smi:%s/origin/%d' % (nowEvent.domain, nowEvent.dbid)
             oevt.preferred_origin_id = o.resource_id
             o.origin_uncertainty = OriginUncertainty()
             o.quality = OriginQuality()
             ou = o.origin_uncertainty
             oq = o.quality
             oevt.creation_info = CreationInfo(version=nowEvent.ver)
             oevt.creation_info.version = "Obspy %s" % __version__
             o.creation_info = CreationInfo(version=nowEvent.ver)
             o.latitude = nowEvent.lat
             o.latitude_errors.uncertainty = kilometer2degrees(nowEvent.herr)
             o.longitude = nowEvent.lon
             o.longitude_errors.uncertainty = kilometer2degrees(nowEvent.herr)
             if nowEvent.depth is not np.nan:
                o.depth = nowEvent.depth * 1e3
                o.depth_errors.uncertainty = nowEvent.verr * 1e3
                o.depth_type = str("from location")
             else:
                o.depth = 9999.0 
                o.depth_errors.uncertainty = 9999.0 * 1e3
                o.depth_type = str("other")
             if nowEvent.fix == '-':
                 o.depth_errors.depth_fixed = True
             else:
                 o.depth_errors.depth_fixed = False
                 if o.depth == 9990.0:
                     o.depth_errors.depth_fixed = True
             o.time = nowEvent.originTime
             if nowEvent.fix == 'O':
                 o.time_fixed=True
             else:
                 o.time_fixed =False
             if nowEvent.fix == 'X' or nowEvent.fix == 'O':
                 o.epicenter_fixed=True
             else:
                 o.epicenter_fixed=False
             o.origin_type=str("hypocenter")
             o.region = str(nowEvent.locRemark)
             ou.horizontal_uncertainty = nowEvent.herr * 1e3
             ou.azimuth_max_horizontal_uncertainty = nowEvent.errAz
             oq.standard_error = nowEvent.err
             oq.azimuthal_gap = nowEvent.gap
             oq.used_phase_count = nowEvent.numPhaTotal
             #oq.used_station_count = count up unique stations in picks
             oq.minimum_distance = nowEvent.dmin
             distances = []

             # Parse each pick
             # Save information as an Arrival to the Origin
             # Save information as a Pick to the Event
             if nowEvent.picks:
                  allPicks = nowEvent.picks
                  # for key,nowPick in allPicks.iteritems():  # iteritems() got omitted in Python3
                  for key,nowPick in allPicks.items():

                       arrival = Arrival()
                       o.arrivals.append(arrival)

                       pick = Pick()
                       oevt.picks.append(pick)  # JJW Is this where this goes?
                       pick.waveform_id = WaveformStreamID(network_code=nowPick.network, station_code=nowPick.station,
                                              location_code=nowPick.location, channel_code=nowPick.channel)

                       # if nowPick.pRes:  # returns False if the value is 0.0 (not ideal)
                       if "P" in nowPick.pRemark:
                           arrival.phase = 'P'
                           arrival.time_residual = nowPick.pRes
                           arrival.time_weight = nowPick.pWeightTrue
                       elif "S" in nowPick.sRemark:
                           arrival.phase = 'S'
                           arrival.time_residual = nowPick.sRes
                           arrival.time_weight = nowPick.sWeightTrue
                       if nowPick.delta:
                           arrival.distance = kilometer2degrees(nowPick.delta)
                           distances.append(nowPick.delta)
                       arrival.azimuth = nowPick.azimuth
                       arrival.takeoff_angle = nowPick.emergenceAngle

                       if "P" in nowPick.pRemark:
                           pick.time = nowPick.pArrival
                           pick.time_errors.uncertainty = nowPick.pRes
                           pick.phase_hint = 'P'
                           # Onset comment
                           if 'I' in nowPick.pRemark:
                                 pick.onset = 'impulsive'
                           elif 'E' in nowPick.pRemark:
                                 pick.onset = 'emergent'
                           else:
                                 pick.onset = 'questionable'
                            # Polarity    
                           if nowPick.pFM == 'U':
                                 pick.polarity = 'positive'
                           elif nowPick.pFM == 'D':
                                 pick.polarity = 'negative'
                           else:
                                 pick.polarity = 'undecidable'
                                
                       elif "S" in nowPick.sRemark:
                           pick.time = nowPick.sArrival
                           pick.time_errors.uncertainty = nowPick.sRes
                           pick.phase_hint = 'S'
                           # Onset comment
                           if 'I' in nowPick.sRemark:
                                pick.onset = 'impulsive'
                           elif 'E' in nowPick.sRemark:
                                pick.onset = 'emergent'
                           else:
                                pick.onset = 'questionable'

                       # print("Done parsing pick.")  # JJW

             if distances:
                  oq.maximum_distance = max(distances)  # get from pick.delta
                  oq.median_distance = median(distances)

             oevt.magnitudes = []
             if nowEvent.magAmp and nowEvent.magAmp != 0.0:
                 ampmag = Magnitude(mag=nowEvent.magAmp, magnitude_type='ML', origin_id=o.resource_id, 
                                    station_count=nowEvent.nAmpMag, comments=Comment(nowEvent.netCodeAmp))
                 oevt.magnitudes.append(ampmag)
                 if nowEvent.magAmp == nowEvent.mag:
                     oevt.preferred_magnitude_id = ampmag.resource_id
             if nowEvent.magCoda and nowEvent.magCoda != 0.0:
                     codamag = Magnitude(mag=nowEvent.magCoda, magnitude_type='Mc', origin_id=o.resource_id,
                                    station_count=nowEvent.nCodaMag, comments=Comment(nowEvent.netCodeDur))
                     oevt.magnitudes.append(codamag)
                     if nowEvent.magCoda == nowEvent.mag:
                         oevt.preferred_magnitude_id = codamag.resource_id
             if nowEvent.magExt and nowEvent.magExt != 0.0:
                 extmag = Magnitude(mag=nowEvent.magExt, magnitude_type=nowEvent.magExtType, origin_id=o.resource_id)
                 oevt.magnitudes.append(extmag)
                 if nowEvent.magExt == nowEvent.mag:
                     oevt.preferred_magnitude_id = extmag.resource_id
             oevt.scope_resource_ids()
             cat.append(oevt)
        return cat
                           
    def writeXML(self, filename, tZone, defaultIconStyle='2' ):
        '''
        method writes catalog to xml file similar to QDM output
        filename=full path and filename to output file
        tZone=timezone of local event
            ex. timezones.Hawaii
                timezones.Alaska
                timezones.Pacific
                (see bottom of timezones.py)
        '''
        iconStyle=defaultIconStyle
        # header
        now=datetime.datetime.now().replace(tzinfo=tZone)
        ltstr=now.strftime('%a %b %d, %Y %H:%M:%S %Z')
        utc=datetime.datetime.utcnow()
        utstr=utc.strftime('%Y%m%d%H%M%S')
        f=open(filename,'w')
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<merge fileTime_loc="%s" fileTime_utc="%s">\n'%(ltstr,utstr))
        self.sortByOriginTime(False) #reverse time sort (stratigraphic order)
        # loop through date
        for i in range(len(self.data)):
            e=self.data[i]
            iconStyle = e.getIcon()
            etime=e.originTime
            secs=etime.second+etime.microsecond*1.e-6
            localtstr=etime.astimezone(tZone).strftime('%a %b %d, %Y %H:%M:%S %Z')
            f.write('<event id="%s"'%e.dbid)
            f.write(' network-code="%s"'%e.domain)
            f.write(' time-stamp="%s" '%now.strftime('%Y/%m/%d_%H:%M:%S'))
            f.write('version="%d">\n'%e.ver)
            f.write('<param name="year" value="%4d"/>\n'%etime.year)
            f.write('<param name="month" value="%02d"/>\n'%etime.month)
            f.write('<param name="day" value="%02d"/>\n'%etime.day)
            f.write('<param name="hour" value="%02d"/>\n'%etime.hour)
            f.write('<param name="minute" value="%02d"/>\n'%etime.minute)
            f.write('<param name="second" value="%5.2f"/>\n'%secs)
            if e.lat:
                f.write('<param name="latitude" value="%.4f"/>\n'%e.lat)
                f.write('<param name="longitude" value="%.4f"/>\n'%e.lon)
            if e.depth:
                f.write('<param name="depth" value="%.2f"/>\n'%e.depth)
            if e.mag:
                f.write('<param name="magnitude" value="%.2f"/>\n'%e.mag)
                f.write('<param name="magnitude-type" value="%s"/>\n'%e.magType)
            numStas = e.numPha=e.numPhaS
            f.write('<param name="num-stations" value="%d"/>\n'%numStas)
            f.write('<param name="num-phases" value="%d"/>\n'%e.numPha)
            f.write('<param name="dist-first-station" value="%.2f"/>\n'%e.dmin)
            f.write('<param name="rms-error" value="%.2f"/>\n'%e.rms)
            f.write('<param name="hor-error" value="%.2f"/>\n'%e.herr)
            f.write('<param name="ver-error" value="%.2f"/>\n'%e.verr)
            f.write('<param name="azimuthal-gap" value="%.2f"/>\n'%e.gap)
            f.write('<param name="local-time" value="%s"/>\n'%localtstr)
            f.write('<param name="icon-style" value="%s"/>\n'%iconStyle)
            f.write('</event>\n')
        f.write('</merge>\n')
        f.close()
        return
    def writeXMLSm(self, filename, tZone, defaultIconStyle='2' ):
        '''
        method writes catalog to xml file similar to QDM output
        filename=full path and filename to output file
        tZone=timezone of local event
            ex. timezones.Hawaii
                timezones.Alaska
                timezones.Pacific
                (see bottom of timezones.py)
        '''
        iconStyle=defaultIconStyle
        # header
        now=datetime.datetime.now().replace(tzinfo=tZone)
        ltstr=now.strftime('%a %b %d, %Y %H:%M:%S %Z')
        utc=datetime.datetime.utcnow().replace(tzinfo=timezones.UTC())
        utstr=utc.strftime('%Y%m%d%H%M%S')
        f=open(filename,'w')
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<merge fileTime_loc="%s" fileTime_utc="%s">\n'%(ltstr,utstr))
        self.sortByOriginTime(False) #reverse time sort (stratigraphic order)
        # loop through date
        for i in range(len(self.data)):
            e=self.data[i]
            iconStyle = e.getIcon()
            etime=e.originTime
            secs=etime.second+etime.microsecond*1.e-6
            localtstr=etime.astimezone(tZone).strftime('%a %b %d, %Y %H:%M:%S %Z')
            f.write('<event id="%s"'%e.dbid)
            f.write(' network-code="%s"'%e.domain)
            f.write(' time-stamp="%s" '%now.strftime('%Y/%m/%d_%H:%M:%S'))
            f.write('version="%d">'%e.ver)
            f.write('<param name="year" value="%4d"/>'%etime.year)
            f.write('<param name="month" value="%02d"/>'%etime.month)
            f.write('<param name="day" value="%02d"/>'%etime.day)
            f.write('<param name="hour" value="%02d"/>'%etime.hour)
            f.write('<param name="minute" value="%02d"/>'%etime.minute)
            f.write('<param name="second" value="%5.2f"/>'%secs)
            if e.lat:
                f.write('<param name="latitude" value="%.4f"/>'%e.lat)
                f.write('<param name="longitude" value="%.4f"/>'%e.lon)
            if e.depth:
                f.write('<param name="depth" value="%.2f"/>'%e.depth)
            if e.mag:
                f.write('<param name="magnitude" value="%.2f"/>'%e.mag)
                f.write('<param name="magnitude-type" value="%s"/>'%e.magType)
            numStas = e.numPha=e.numPhaS
            f.write('<param name="num-stations" value="%d"/>'%numStas)
            f.write('<param name="num-phases" value="%d"/>'%e.numPha)
            f.write('<param name="dist-first-station" value="%.2f"/>'%e.dmin)
            f.write('<param name="rms-error" value="%.2f"/>'%e.rms)
            f.write('<param name="hor-error" value="%.2f"/>'%e.herr)
            f.write('<param name="ver-error" value="%.2f"/>'%e.verr)
            f.write('<param name="azimuthal-gap" value="%.2f"/>'%e.gap)
            f.write('<param name="local-time" value="%s"/>'%localtstr)
            f.write('<param name="icon-style" value="%s"/>'%iconStyle)
            f.write('</event>\n')
        f.write('</merge>\n')
        f.close()
        return       
    #container methods
    def __getitem__(self,i):
        return self.data.__getitem__(i)
    def __setitem__(self,i,d):
        if d.originTime<self.starttime: self.starttime=d.originTime
        if d.originTime>self.endtime: self.endtime=d.originTime
        self.data[i]=d
        return
    def __delitem__(self,i):
        self.data.__delitem__(i)
        return
    def __len__(self):
        return self.data.__len__()
    def __contains__(self,d):
        return self.data.__contains__(d)
    def __iter__(self):
        return self.data.__iter__()
    '''
    def __str__(self,stream=None):
        stream = stream or sys.stderr            
        stream.write('Start Time = %s\n' % self.starttime)
        stream.write('End Time = %s\n' % self.endtime)
        stream.write('Number of Earthquakes = %s\n' % len(self.data))
        if self.starttime:
            return 'Initalized catalog object\n'
        else:
            return 'Populated catalog object\n'
    '''
    def append(self,d):
        if not self.starttime or d.originTime<self.starttime:
            self.starttime=d.originTime
        if not self.endtime or d.originTime>self.endtime: 
            self.endtime=d.originTime
        self.data.append(d)
        return

def main():
    basedir = '/Users/wthelen/Documents/hawaii/science/sumfiles/'
    filesCat = ['sum1960s', 'sum1970s', 'sum1980s', 'sum1990s','sum2k'] 
    filesAQMS = ['aqms_20090401_20150603.sum']
    
    cat = hypoCatalog()
    for f in filesCat:
        fname = '%s%s' % (basedir,f)
        cat.readSumFile(fname)
    cat.writeNEIC(filename='hvoCatalog_1959_2009.csv')
    
    catAQMS = hypoCatalog()
    catAQMS.readSumFile('%s%s'% (basedir, filesAQMS[0]))
    catAQMS.writeNEIC(filename='hvoCatalog_2009_2015.csv')

if __name__ == "__main__":
    main()