#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dependencies:
A UNIX like shell must be available
pip install (--user) numpy
pip install (--user) editdistance
pip install (--user) Polygon2
pip install (--user) Pillow (if cv2 is not available)
"""
import numpy.matlib
import numpy as np
from StringIO import StringIO
import tarfile
import zipfile
#from commands import getoutput as go #fast access to the shell
import re
try:
    import cv2
    imwrite=cv2.imwrite
    imread=cv2.imread
except:
    #if opencv is not available but PIL is it can alternatively be used for IO
    import PIL as pil
    imwrite=lambda filename,imgArray:pil.Image.fromarray(imgArray).save(filename)
    imread= lambda filename:numpy.array(pil.Image.open(filename))



def convLTRB24point(ltbr):
    """
    Converts Rectagles defined by the top left corner, and the bottom-right 
    corner to a quadrilateral defined by 4 points in clockwise order. If more 
    than 4 columns are passed as input, the will be appended after the 8th 
    column of the returned matrix.
    
    Args:
        ltbr: An numpy matrix whos columns are the x coordinate of the top-left 
        corner, the y coordinate of the top-left corner, the x coordinate of 
        the bottom-right corner, and the y coordinate of bottom-right corner.

    Returns:
        A numpy array where each column is a coordinate of one of the 
        quadrilateral's points in the order x1,y1,x2,y2,x3,y3,x4,y4.
    """
    L=ltbr[:,0]
    T=ltbr[:,1]
    B=ltbr[:,2]
    R=ltbr[:,3]
    res = np.concatenate([L.reshape([-1,1]),T.reshape([-1,1]),R.reshape([-1,1]),T.reshape([-1,1]),R.reshape([-1,1]),B.reshape([-1,1]),L.reshape([-1,1]),B.reshape([-1,1])],axis=1)
    if ltbr.shape[1]>4:
        res=np.concatenate([res,ltbr[:,4:]])
    return res


def conv4pointToLTBR(pointMat):
    """
    Converts quadrilaterals to the smallest axis-aligned rectangles defined by
    the top left  and the bottom-right corners. If more than 8 columns are 
    passed as input, the will be appended after the 4th column of the returned 
    matrix.
    
    Args:
        pointMat: A numpy array where each column is a coordinate of one of the 
        quadrilateral's points in the order x1,y1,x2,y2,x3,y3,x4,y4.

    Returns:
        An numpy matrix whos columns are the x coordinate of the top-left 
        corner, the y coordinate of the top-left corner,  the x coordinate of 
        the bottom-right corner, and the y coordinate of bottom-right corner.
    """
    L=pointMat[:,[0,2,4,6]].min(axis=1)
    R=pointMat[:,[0,2,4,6]].max(axis=1)
    T=pointMat[:,[1,3,5,7]].min(axis=1)
    B=pointMat[:,[1,3,5,7]].max(axis=1)
    res = np.concatenate([L.reshape([-1,1]),T.reshape([-1,1]),R.reshape([-1,1]),B.reshape([-1,1])],axis=1)
    if pointMat.shape[1]>8:
        res=np.concatenate([res,pointMat[:,8:]])
    return res


def convLTWH2LTBR(ltwh):
    L=ltwh[:,0]
    R=ltwh[:,2]+1+L
    T=ltwh[:,1]
    B=ltwh[:,3]+1+T
    res = np.concatenate([L.reshape([-1,1]),T.reshape([-1,1]),R.reshape([-1,1]),B.reshape([-1,1])],axis=1)
    if ltwh.shape[1]>4:
        res=np.concatenate([res,ltwh[:,8:]])
    return res


def convLTRB2LTWH(ltrb):
    L=ltrb[:,0]
    T=ltrb[:,1]
    W=1+(ltrb[:,2]-ltrb[:,0])
    H=1+(ltrb[:,3]-ltrb[:,1])
    res = np.concatenate([L.reshape([-1,1]),T.reshape([-1,1]),W.reshape([-1,1]),H.reshape([-1,1])],axis=1)
    if ltrb.shape[1]>4:
        res=np.concatenate([res,ltrb[:,4:]])
    return res

def loadBBoxTranscription(fileData,**kwargs):
    txt=fileData
    txt=re.sub(r'[^\x00-\x7f]',r'',txt)#removing the magical bytes crap
    if txt=='':
        return np.empty([0,8]),[]
    lines=[l.strip().split(',') for l in txt.split('\n') if (len(l.strip())>0)]
    colFound=min([len(l) for l in lines])-1
    if colFound>=4 and colFound<8:#THIS is ugly 
        resBoxes=np.empty([len(lines),4],dtype='int32')
        resTranscriptions=np.empty(len(lines), dtype=object)
        for k in range(len(lines)):
            resBoxes[k,:]=[int(float(c)) for c in lines[k][:4]]
            resTranscriptions[k]=','.join(lines[k][4:])
    elif colFound>=8:#THIS is ugly as well
        resBoxes=np.empty([len(lines),8],dtype='int32')
        resTranscriptions=np.empty(len(lines), dtype=object)
        for k in range(len(lines)):
            resBoxes[k,:]=[int(float(c)) for c in lines[k][:8]]
            resTranscriptions[k]=','.join(lines[k][8:])
    else:
        raise Exception('Wrong columns found')
    return (resBoxes,resTranscriptions)

def loadArchiveAsFiledataDict(fileExtention,archiveFdata=None):
    if archiveFdata is None:
        archiveFdata=open(fileExtention).read()
    if fileExtention.lower().endswith('tar.gz'):
        fd=StringIO(str(archiveFdata))
        archive=tarfile.open(None,'r',fileobj=fd)
        return dict([(m.name,archive.extractfile(m).read()) for m in archive.getmembers() if m.isfile()])
    elif fileExtention.lower().endswith('zip'):
        fd=StringIO(str(archiveFdata))
        archive=zipfile.ZipFile(fd,'r')
        return dict([(name,archive.read(name)) for name in archive.namelist() if name[-1]!='/'])
    else:
        raise Exception('loadArchiveAsFiledataDict: unknown fileExtention :'+str(fileExtention))

def packFiledataDicts(*dictionaryList):
    fname2id=lambda x:x.split('/')[-1].split('.')[0]
    idDictList=[]
    for fileDict in dictionaryList:
        idDictList.append({fname2id(k):fileDict[k] for k in fileDict.keys()})
    return zip(*[[d[k] for k in sorted(d.keys())] for d in idDictList])
