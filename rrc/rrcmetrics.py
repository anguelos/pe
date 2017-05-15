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
import re
import sys
#import time
#import json
#import os
#from commands import getoutput as go #fast access to the shell

import Polygon as plg
import editdistance

import rrcio

def getPixelIoU(gtImg,submImg):
    #TODO TEST THOROUGHLY
    def compress(img):
        intImg=np.empty(img.shape[:2],dtype='int32')
        if len(img.shape)==3:
            intImg[:,:]=img[:,:,0]
            intImg[:,:]+=(256*img[:,:,1])
            intImg[:,:]+=((256**2)*img[:,:,1])
        else:
            intImg[:,:]=img[:,:]
        un=np.unique(intImg)
        idx=np.zeros(un.max()+1)
        idx[un]=np.arange(un.shape[0],dtype='int32')
        return idx[intImg],un.max()+1
    if gtImg.shape[:2]!=submImg[:2]:
        raise Exception("gtImg and submImg must have the same size")
    gt,maxGt=compress(gtImg)
    subm,maxSubm=compress(gtImg)
    comb=gt*maxSubm+subm
    intMatrix=np.bincount(comb.reshape(-1)).reshape([maxSubm,maxGt])
    uMatrix=np.zeros(intMatrix.shape)
    uMatrix[:,:]+=intMatrix.sum(axis=0)[None,:]
    uMatrix[:,:]+=intMatrix.sum(axis=1)[:,None]
    uMatrix-=intMatrix    
    return intMatrix/uMatrix.astype('float64'),intMatrix,uMatrix


def get4pointIoU(gtMat,sampleMat):
    e=.0000000000000001
    iMat=np.zeros([gtMat.shape[0],sampleMat.shape[0]])
    uMat=np.zeros([gtMat.shape[0],sampleMat.shape[0]])
    gtAreas=np.zeros(gtMat.shape[0])
    gtPolList=[]
    gtAreas=np.zeros(gtMat.shape[0])
    for gtPolNum in range(gtMat.shape[0]):
        gtPolList.append(plg.Polygon(gtMat[gtPolNum,:].reshape([4,2])))
        gtAreas[gtPolNum]=gtPolList[-1].area()
    samplePolList=[]
    sampleAreas=np.zeros(sampleMat.shape[0])
    for samplePolNum in range(sampleMat.shape[0]):
        samplePolList.append(plg.Polygon(sampleMat[samplePolNum,:].reshape([4,2])))
        sampleAreas[samplePolNum]=samplePolList[-1].area()
    for submPolNum in range(sampleMat.shape[0]):
        for gtPolNum in range(gtMat.shape[0]):
            iMat[gtPolNum,submPolNum]=(gtPolList[gtPolNum]&(samplePolList[submPolNum])).area()
            uMat[gtPolNum,submPolNum]=(gtPolList[gtPolNum]|(samplePolList[submPolNum])).area()
    return [iMat/(uMat+e),iMat,uMat]


def getEditDistanceMat(gtTranscriptions,sampleTranscriptions):
    outputShape=[len(gtTranscriptions),len(sampleTranscriptions)]
    distMat=np.empty(outputShape)
    maxSizeMat=np.empty(outputShape)
    for gtNum in range(len(gtTranscriptions)):
        for sampleNum in range(len(sampleTranscriptions)):
            distMat[gtNum,sampleNum]=editdistance.eval(gtTranscriptions[gtNum],sampleTranscriptions[sampleNum])
            maxSizeMat[gtNum,sampleNum]=max(len(gtTranscriptions[gtNum]),len(sampleTranscriptions[sampleNum]))
    return distMat/maxSizeMat,distMat


def maskNonMaximalIoU(IoU,axis=1):
    """Generates a mask so that multiple recognitions of 
    can be removed from an IoU matrix.
    
    Args:
        IoU: a matrix where each row is a groundtrouth object and each column
        a retrived object, the cells contain the IoU ratio of objects
        axis: whether columns or rows should be masked.
    
    Returns: a matrix with ones only on maximum cells either column wise or row
    wise.
    """
    if IoU.shape[0]==0 or IoU.shape[1]==0:
        return np.zeros(IoU.shape)
    res=np.zeros(IoU.shape)
    if axis==0:
        res[np.arange(res.shape[0],dtype='int32'),IoU.argmax(axis=1)]=1
    elif axis==1:
        res[IoU.argmax(axis=0),np.arange(res.shape[1],dtype='int32')]=1
    else:
        raise Exception('Axis must be 0 or 1')
    return res


def get2PointIoU(gtMatLTWH,subMatLTWH,suppresAxis=[1]):
    #gtMat=convLTRB2LTWH(gtMat)
    #resMat=convLTRB2LTWH(resMat)
    gtMat=gtMatLTWH
    subMat=subMatLTWH
    gtLeft=numpy.matlib.repmat(gtMat[:,0],subMat.shape[0],1)
    gtTop=numpy.matlib.repmat(gtMat[:,1],subMat.shape[0],1)
    gtRight=numpy.matlib.repmat(gtMat[:,0]+gtMat[:,2]-1,subMat.shape[0],1)
    gtBottom=numpy.matlib.repmat(gtMat[:,1]+gtMat[:,3]-1,subMat.shape[0],1)
    gtWidth=numpy.matlib.repmat(gtMat[:,2],subMat.shape[0],1)
    gtHeight=numpy.matlib.repmat(gtMat[:,3],subMat.shape[0],1)
    resLeft=numpy.matlib.repmat(subMat[:,0],gtMat.shape[0],1).T
    resTop=numpy.matlib.repmat(subMat[:,1],gtMat.shape[0],1).T
    resRight=numpy.matlib.repmat(subMat[:,0]+subMat[:,2]-1,gtMat.shape[0],1).T
    resBottom=numpy.matlib.repmat(subMat[:,1]+subMat[:,3]-1,gtMat.shape[0],1).T
    resWidth=numpy.matlib.repmat(subMat[:,2],gtMat.shape[0],1).T
    resHeight=numpy.matlib.repmat(subMat[:,3],gtMat.shape[0],1).T
    intL=np.max([resLeft,gtLeft],axis=0)
    intT=np.max([resTop,gtTop],axis=0)
    intR=np.min([resRight,gtRight],axis=0)
    intB=np.min([resBottom,gtBottom],axis=0)
    intW=(intR-intL)+1
    intW[intW<0]=0
    intH=(intB-intT)+1
    intH[intH<0]=0
    #TODO fix the following transpose
    I=(intH*intW).T
    U=resWidth.T*resHeight.T+gtWidth.T*gtHeight.T-I
    IoU=I/(U+.0000000001)
    return (IoU,I,U)


def filterDontCares(IoU,edDist,gtTrans,dontCare):
    """Removes rows and columns from a 
    """
    if edDist is None:
        edDist=np.empty(IoU.shape)
    #if IoU.shape[0]==0 or IoU.shape[1]==0:
    #    return IoU,edDist
    removeGt=np.where(gtTrans==dontCare)[0].tolist()
    highestIoUPos=np.argmax(IoU,axis=0)
    removeSubm=[k for k in range(IoU.shape[1]) if (highestIoUPos[k] in removeGt)]
    IoU=np.delete(np.delete(IoU,removeSubm,axis=1),removeGt,axis=0)
    edDist=np.delete(np.delete(edDist,removeSubm,axis=1),removeGt,axis=0)
    return IoU,edDist


def get4pEndToEndMetric(gtSubmFdataTuples,**kwargs):
    e=.00000000000001
    p={'dontCare':'###','iouThr':.5,'maxEdist':0,'caseInsencitive':True,'ignoreChars':"[@!?\.,%]|\'[sS]"}
    p.update(kwargs)
    allRelevant=0
    allRetrieved=0
    correct=0
    for gtStr,submStr in gtSubmFdataTuples:
        gtLoc,gtTrans=rrcio.loadBBoxTranscription(gtStr)
        submLoc,submTrans=rrcio.loadBBoxTranscription(submStr)
        if p['caseInsencitive']:
            gtTrans=[t.lower() for t in gtTrans]
            submTrans=[t.lower() for t in submTrans]
        if p['ignoreChars']:
            punctuationRe=re.compile(p['ignoreChars'])
            gtTrans=[punctuationRe.sub('',t) for t in gtTrans]
            submTrans=[punctuationRe.sub('',s) for s in submTrans]
        gtTrans=np.array(gtTrans,dtype='object')
        submTrans=np.array(submTrans,dtype='object')
        IoU=get4pointIoU(gtLoc,submLoc)[0]
        edDist=getEditDistanceMat(gtTrans,submTrans)[0]
        if p['dontCare']!='':
            IoU,edDist=filterDontCares(IoU,edDist,gtTrans,p['dontCare'])
        allRelevant+=IoU.shape[0]
        allRetrieved+=IoU.shape[1]
        correct+=np.sum((IoU>=p['iouThr'])*(edDist<=p['maxEdist'])*maskNonMaximalIoU(IoU,1))
    precision=float(correct)/(allRetrieved+e)
    recall=float(correct)/(allRelevant+e)
    FM=(2*precision*recall)/(precision+recall+e)
    return FM,precision,recall


def get2pEndToEndMetric(gtSubmFdataTuples,**kwargs):
    e=.00000000000001
    p={'dontCare':'###','iouThr':.5,'maxEdist':0}
    p.update(kwargs)
    allRelevant=0
    allRetrieved=0
    correct=0
    for gtStr,submStr in gtSubmFdataTuples:
        gtLoc,gtTrans=rrcio.loadBBoxTranscription(gtStr)
        submLoc,submTrans=rrcio.loadBBoxTranscription(submStr)
        if gtLoc.shape[1]==8:
            gtLoc=rrcio.conv4pointToLTBR(gtLoc)
        if submLoc.shape[1]==8:
            submLoc=rrcio.conv4pointToLTBR(submLoc)
        IoU=get2PointIoU(gtLoc,submLoc)[0]
        edDist=getEditDistanceMat(gtTrans,submTrans)[0]
        if p['dontCare']!='':
            IoU,edDist=filterDontCares(IoU,edDist,gtTrans,p['dontCare'])
        allRelevant+=IoU.shape[0]
        allRetrieved+=IoU.shape[1]
        correct+=np.sum((IoU>=p['iouThr'])*(edDist<=p['maxEdist'])*maskNonMaximalIoU(IoU,1))
    precision=float(correct)/(allRetrieved+e)
    recall=float(correct)/(allRelevant+e)
    FM=(2*precision*recall)/(precision+recall+e)
    return FM,precision,recall


def getFSNSMetrics(gtIdTransDict,methodIdTransDict):
    """Provides metrics for the FSNS dataset. 
    FM, precision, recall and correctSequences are an implementation of the metrics described in
    "End-to-End Interpretation of the French Street Name Signs Dataset"
    [https://link.springer.com/chapter/10.1007%2F978-3-319-46604-0_30]


    Params:
        gtIdTransDict : sample_id to data dictionary. A simple file name to file contents might do.
        methodIdTransDict : sample_id to data dictionary. A simple file name to file contents might do.
    
    returns:
        A tuple with floats between 0 and 1 with all worth reporting measurements.
        FM, Precision, Recall, global correct word trascriptions, if someone returned 
        "rue" as the transcription of every image, assuming half the images have it, he 
        would get a precision of 50%, a recall of ~5% and an FM of ~9.1%.
        He would get a correctSequences score of 0%, and a similarity of e%.
    """
    def compareTexts(sampleTxt,gtTxt):
        relevant=gtTxt.lower().split()
        retrieved=sampleTxt.lower().split()
        correct=(set(relevant).intersection(set(retrieved)))
        similarity=1.0/(1+editdistance.eval(gtTxt.lower(),sampleTxt.lower()))
        res=(len(correct),len(relevant),len(retrieved),relevant!=retrieved,similarity)
        return res
    mDict={k:'' for k in gtIdTransDict.keys()}
    mDict.update(methodIdTransDict)
    methodIdTransDict=mDict
    methodKeys=methodIdTransDict.keys()
    gtKeys=gtIdTransDict.keys()
    if len(methodKeys)!= len(set(methodKeys))  or len(gtKeys)!= len(set(gtKeys)) or len(set(methodKeys)-set(gtKeys))>0 :#gt and method dissagree on samples
        sys.stderr.write("GT and submission dissagree on the sample ids\n")
        sys.exit(1)
    corectRelevantRetrievedSimilarity=np.zeros([len(gtKeys),5],dtype='float32')
    for k in range(len(gtKeys)):
        sId=gtKeys[k]
        corectRelevantRetrievedSimilarity[k,:]=compareTexts(methodIdTransDict[sId],gtIdTransDict[sId])
    precision=(corectRelevantRetrieved[:,0].sum()/(corectRelevantRetrieved[:,1].sum()))
    recall=(corectRelevantRetrieved[:,0].sum()/(corectRelevantRetrieved[:,2].sum()))
    FM=(2*precision*recall)/(precision+recall)
    correctSequences=corectRelevantRetrieved[:,3].mean()
    similarity=corectRelevantRetrieved[:,4].mean()
    combinedSoftMetric=(1-FM)*FM+FM*similarity#The better FM is, the less it maters in the overall score
    return combinedSoftMetric,FM,precision,recall,similarity,correctSequences
