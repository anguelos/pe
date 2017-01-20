#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Dependencies:
pip install (--user) numpy
pip install (--user) editdistance
pip install (--user) Polygon2
pip install (--user) Pillow (if cv2 is not available)
"""
import numpy.matlib
import numpy as np
import re
import sys
import time
import json
import os
from commands import getoutput as go #fast access to the shell

try:
    import cv2
    imwrite=cv2.imwrite
    imread=cv2.imread
except:
    #if opencv is not available but PIL is it can alternatively be used for IO
    import PIL as pil
    imwrite=lambda filename,imgArray:pil.Image.fromarray(imgArray).save(filename)
    imread= lambda filename:numpy.array(pil.Image.open(filename))

import Polygon as plg
import editdistance


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

    
def getFSNSMetrics(gtIdTransDict,methodIdTransDict):
    def compareTexts(sampleTxt,gtTxt):
        relevant=gtTxt.lower().split()
        retrieved=sampleTxt.lower().split()
        correct=(set(relevant).intersection(set(retrieved)))
        res=(len(correct),len(relevant),len(retrieved),relevant!=retrieved)
        return res
    mDict={k:'' for k in gtIdTransDict.keys()}
    mDict.update(methodIdTransDict)
    methodIdTransDict=mDict
    methodKeys=methodIdTransDict.keys()
    gtKeys=gtIdTransDict.keys()
    if len(methodKeys)!= len(set(methodKeys))  or len(gtKeys)!= len(set(gtKeys)) or len(set(methodKeys)-set(gtKeys))>0 :#gt and method dissagree on samples
        sys.stderr.write("GT and submission dissagree on the sample ids\n")
        sys.exit(1)
    corectRelevantRetrieved=np.zeros([len(gtKeys),4])
    for k in range(len(gtKeys)):
        sId=gtKeys[k]
        corectRelevantRetrieved[k,:]=compareTexts(methodIdTransDict[sId],gtIdTransDict[sId])
    precision=(corectRelevantRetrieved[:,0].sum()/(corectRelevantRetrieved[:,1].sum()))
    recall=(corectRelevantRetrieved[:,0].sum()/(corectRelevantRetrieved[:,2].sum()))
    FM=(2*precision*recall)/(precision+recall)
    return FM,precision,recall,float(corectRelevantRetrieved[:,3].mean()),corectRelevantRetrieved


def getPixelIoU(gtImg,submImg):
    #TODO TEST
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
    iMat=np.zeros([gtMat.shape[0],sampleMat.shape[0]])
    uMat=np.zeros([gtMat.shape[0],sampleMat.shape[0]])
    gtAreas=np.zeros(gtMat.shape[0])
    gtPolList=[]
    gtAreas=np.zeros(gtMat.shape[0])
    for gtPolNum in range(gtMat.shape[0]):
        gtPolList.append(plg.Polygon(gtMat[gtPolNum].reshape([2,-1]).T))
        gtAreas[gtPolNum]=gtPolList[-1].area()
    samplePolList=[]
    sampleAreas=np.zeros(sampleMat.shape[0])
    for samplePolNum in range(sampleMat.shape[0]):
        samplePolList.append(plg.Polygon(sampleMat[samplePolNum].reshape([2,-1]).T))
        sampleAreas[samplePolNum]=samplePolList[-1].area()
    for submPolNum in range(sampleMat.shape[0]):
        for gtPolNum in range(gtMat.shape[0]):
            iMat[gtPolNum,submPolNum]=(samplePolList[submPolNum]&(samplePolList[submPolNum])).area()
            uMat[gtPolNum,submPolNum]=(samplePolList[submPolNum]|(samplePolList[submPolNum])).area()
    return [iMat/(uMat+.0000000000000001),iMat,uMat]


def getEditDistanceMat(gtTranscriptions,sampleTranscriptions):
    outputShape=[len(gtTranscriptions),len(sampleTranscriptions)]
    distMat=np.empty(outputShape)
    maxSizeMat=np.empty(outputShape)
    for gtNum in range(len(gtTranscriptions)):
        for sampleNum in range(len(sampleTranscriptions)):
            distMat[gtNum,sampleNum]=editdistance.eval(gtTranscriptions[gtNum],sampleTranscriptions[sampleNum])
            maxSizeMat[gtNum,sampleNum]=max(len(gtTranscriptions[gtNum]),len(sampleTranscriptions[sampleNum]))
    return distMat/maxSizeMat,distMat


def get2PointIoU(gtMat,resMat):
    gtMat=convLTRB2LTWH(gtMat)
    resMat=convLTRB2LTWH(resMat)
    gtLeft=numpy.matlib.repmat(gtMat[:,0],resMat.shape[0],1)
    gtTop=numpy.matlib.repmat(gtMat[:,1],resMat.shape[0],1)
    gtRight=numpy.matlib.repmat(gtMat[:,0]+gtMat[:,2]-1,resMat.shape[0],1)
    gtBottom=numpy.matlib.repmat(gtMat[:,1]+gtMat[:,3]-1,resMat.shape[0],1)
    gtWidth=numpy.matlib.repmat(gtMat[:,2],resMat.shape[0],1)
    gtHeight=numpy.matlib.repmat(gtMat[:,3],resMat.shape[0],1)
    resLeft=numpy.matlib.repmat(resMat[:,0],gtMat.shape[0],1).T
    resTop=numpy.matlib.repmat(resMat[:,1],gtMat.shape[0],1).T
    resRight=numpy.matlib.repmat(resMat[:,0]+resMat[:,2]-1,gtMat.shape[0],1).T
    resBottom=numpy.matlib.repmat(resMat[:,1]+resMat[:,3]-1,gtMat.shape[0],1).T
    resWidth=numpy.matlib.repmat(resMat[:,2],gtMat.shape[0],1).T
    resHeight=numpy.matlib.repmat(resMat[:,3],gtMat.shape[0],1).T
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


def loadBBoxTranscription(fname,**kwargs):
    txt=open(fname).read()
    txt=re.sub(r'[^\x00-\x7f]',r'',txt)#removing the magical bytes crap
    lines=[l.strip().split(',') for l in txt.split('\n') if (len(l.strip())>0)]
    colFound=min([len(l) for l in lines])-1
    if colFound==4:
        resBoxes=np.empty([len(lines),4],dtype='int32')
        resTranscriptions=np.empty(len(lines), dtype=object)
        for k in range(len(lines)):
            resBoxes[k,:]=[int(c) for c in lines[k][:4]]
            resTranscriptions[k]=','.join(lines[k][4:])
    elif colFound==8:
        resBoxes=np.empty([len(lines),8],dtype='int32')
        resTranscriptions=np.empty(len(lines), dtype=object)
        for k in range(len(lines)):
            resBoxes[k,:]=[int(c) for c in lines[k][:8]]
            resTranscriptions[k]=','.join(lines[k][8:])
    else:
        sys.stderr.write('Cols found '+str(colFound)+'\n')
        sys.stderr.flush()
        raise Exception('Wrong columns found')
    return (resBoxes,resTranscriptions)

    
def get4pEndToEndMetric(gtSubmFnameTuples):
    allRelevant=0
    allRetrieved=0
    correct=0
    for gtFname,submFname in gtSubmFnameTuples:
        gtLoc,gtTrans=loadBBoxTranscription(gtFname)
        submLoc,submTrans=loadBBoxTranscription(submFname)
        IoU=get4pointIoU(gtLoc,submLoc)[0]
        edDist=getEditDistanceMat(gtTrans,submTrans)[0]
        allRelevant+=gtLoc.shape[0]
        allRetrieved+=submLoc.shape[0]
        correct+=np.sum((IoU>=.5)*(edDist==0))
    precision=float(correct)/allRetrieved
    recall=float(correct)/allRelevant
    FM=(2*precision*recall)/(precision+recall+.00000000000001)
    return FM,precision,recall

    
def get2pEndToEndMetric(gtSubmFnameTuples):
    #TODO remove from evaluation as is also visualisation
    allRelevant=0
    allRetrieved=0
    correct=0
    for gtFname,submFname in gtSubmFnameTuples:
        gtLoc,gtTrans=loadBBoxTranscription(gtFname)
        submLoc,submTrans=loadBBoxTranscription(submFname)
        if gtLoc.shape[1]==8:
            gtLoc=conv4pointToLTBR(gtLoc)
        if submLoc.shape[1]==8:
            submLoc=conv4pointToLTBR(submLoc)
        IoU=get2PointIoU(gtLoc,submLoc)[0]
        edDist=getEditDistanceMat(gtTrans,submTrans)[0]
        allRelevant+=gtLoc.shape[0]
        allRetrieved+=submLoc.shape[0]
        correct+=np.sum((IoU>=.5)*(edDist==0))
    precision=float(correct)/allRetrieved
    recall=float(correct)/allRelevant
    FM=(2*precision*recall)/(precision+recall+.00000000000001)
    return FM,precision,recall

    
def plotRectangles(rects,transcriptions,bgrImg,rgbCol):
    bgrCol=np.array(rgbCol)[[2,1,0]]
    res=bgrImg.copy()
    pts=np.empty([rects.shape[0],5,1,2])
    if rects.shape[1]==4:
        x=rects[:,[0,2,2,0,0]]
        y=rects[:,[1,1,3,3,1]]
    elif rects.shape[1]==8:
        x=rects[:,[0,2,4,6,0]]
        y=rects[:,[1,3,5,7,1]]
    else:
        raise Exception()
    pts[:,:,0,0]=x
    pts[:,:,0,1]=y
    pts=pts.astype('int32')
    ptList=[pts[k,:,:,:] for k in range(pts.shape[0])]
    if not (transcriptions is None):
        for rectNum in range(rects.shape[0]):
            res=cv2.putText(res,transcriptions[rectNum],(rects[rectNum,0],rects[rectNum,1]),1,cv2.FONT_HERSHEY_PLAIN,bgrCol)
    res=cv2.polylines(res,ptList,False,bgrCol)
    return res


def getReport(imgGtSubmFnames,**kwargs):
    """imgGtSubmFnames is a list of tuples with three strings:
       The first one is the path to the input image
       The second is the path to the 4point+transcription Gt
       The third is the path to the 4point+transcription Solution
    """
    p={'IoUthreshold':.5,'dontCare':'###','visualise':True,'outReportDir':'/tmp/report/'}
    p.update(kwargs)
    allIoU=[]
    allEqual=[]
    allCare=[]
    accDict={}
    if os.path.exists(p['outReportDir']):
        raise Exception('Output directory must no exist')
    go('mkdir -p '+p['outReportDir'])
    startTime=time.time()
    for inImgFname,gtFname,submFname in imgGtSubmFnames:
        sampleFname=p['outReportDir']+'/'+inImgFname.split('/')[-1].split('.')[0]
        rectSubm,transcrSubm=loadBBoxTranscription(submFname)
        rectGt,transcrGt=loadBBoxTranscription(gtFname)
        rectGt=conv4pointToLTBR(rectGt)
        rectSubm=conv4pointToLTBR(rectSubm)
        IoU=get2PointIoU(rectGt,rectSubm)[0]
        strEqual=np.zeros([len(transcrSubm),len(transcrGt)])
        strCare=np.zeros([len(transcrSubm),len(transcrGt)])
        for gt in range(transcrGt.shape[0]):
            strCare[:,gt]=(transcrGt[gt]!=p['dontCare'])
            for subm in range(transcrSubm.shape[0]):
                strEqual[subm,gt]=(transcrGt[gt]==transcrSubm[subm])
        IoU=(IoU>p['IoUthreshold']).astype('float')
        allIoU.append(IoU)
        allEqual.append(strEqual)
        allCare.append(strCare)
        img=imread(inImgFname)
        if p['visualise']:
            img=plotRectangles(rectGt,transcrGt,img,[0,255,0])
            img=plotRectangles(rectSubm,transcrSubm,img,[255,0,0])
            imwrite(sampleFname+'.png',img)
        else:
            imwrite(sampleFname+'.png',img)
        resTbl='<table border=1>\n<tr><td></td><td>'
        resTbl+='</td> <td>'.join([s for s in transcrGt])+'</tb></tr>\n'
        for k in range(IoU.shape[0]):
            resTbl+='<tr><td>'+transcrSubm[k]+'</td><td>'
            resTbl+='</td><td>'.join([str(int(k*10000)/100.0) for k in IoU[k,:]*strEqual[k,:]])+'</td></tr>\n'
        resTbl+='</table>\n'
        resHtml='<html><body>\n<h3>'+inImgFname.split('/')[-1].split('.')[0]+'</h3>\n'
        
        acc=((IoU*strEqual).sum()/float(IoU.shape[1]))
        if p['dontCare']!='':
            precision=(IoU*strEqual).max(axis=1)[(strCare*IoU).sum(axis=1)>0].mean()
            if np.isnan(precision):
                precision=0
            recall=(IoU*strEqual).max(axis=0)[(strCare).sum(axis=0)>0].mean()
            if np.isnan(recall):
                recall=0
        else:
            precision=(IoU*strEqual).max(axis=1).mean()
            recall=(IoU*strEqual).max(axis=0).mean()
        fm=(2.0*precision*recall)/(.0000001+precision+recall)
        accDict[sampleFname+'.html']=[acc,precision,recall,fm]
        resHtml+='<hr>\n<table><tr>'
        resHtml+='<td>Accuracy : '+str(int(acc*10000)/100.0)+'% </td>'
        resHtml+='<td>Precision : '+str(int(precision*10000)/100.0)+'% </td>'
        resHtml+='<td>Recall : '+str(int(recall*10000)/100.0)+'% </td>'
        resHtml+='<td> FM : '+str(int(fm*10000)/100.0)+'% </td>'
        resHtml+='</tr></table>\n<hr>\n'
        resHtml+='<img src="'+sampleFname+'.png"/>\n<hr>\n'+resTbl
        resHtml+='</body></html>'
        open(sampleFname+'.html','w').write(resHtml)
    gtSize=sum([iou.shape[1] for iou in allIoU])
    submSize=sum([iou.shape[0] for iou in allIoU])
    IoU=np.zeros([submSize,gtSize])
    strEqual=np.zeros([submSize,gtSize])
    strCare=np.zeros([submSize,gtSize])
    gtIdx=0
    submIdx=0
    for k in range(len(allIoU)):
        submSize,gtSize=allIoU[k].shape
        IoU[submIdx:submIdx+submSize,gtIdx:gtIdx+gtSize]=allIoU[k]
        strEqual[submIdx:submIdx+submSize,gtIdx:gtIdx+gtSize]=allEqual[k]
        strCare[submIdx:submIdx+submSize,gtIdx:gtIdx+gtSize]=allCare[k]
        gtIdx+=gtSize
        submIdx+=submSize
    acc=((IoU*strEqual).sum()/float(IoU.shape[1]))
    if p['dontCare']!='':
        precision=(IoU*strEqual).max(axis=1)[(IoU*strCare).sum(axis=1)>0].mean()
        recall=(IoU*strEqual).max(axis=0)[(strCare).sum(axis=0)>0].mean()
    else:
        precision=(IoU*strEqual).max(axis=1).mean()
        recall=(IoU*strEqual).max(axis=0).mean()
    fm=(2.0*precision*recall)/(.0000001+precision+recall)
    resHtml='<body><html>\n<h3>Report on end 2 end</h3>\n'
    resHtml+='<hr>\n<table border=1>'
    resHtml+='<tr><td> Total Samples: </td><td>'+str(IoU.shape[1])+'</td></tr>'
    resHtml+='<tr><td> Detected Samples : </td><td>'+str(IoU.shape[0])+' </td></tr>'
    resHtml+='<tr><td>Correct Samples : </td><td>'+str(int((IoU*strEqual).sum()))+' </td></tr>'
    resHtml+='<tr><td>Computation Time : </td><td>'+str(int(1000*(time.time()-startTime))/1000.0)+' sec. </td></tr>'
    resHtml+='<tr><td></td><td></td></tr>\n'
    resHtml+='<tr><td>Accuracy : </td><td>'+str(int(acc*10000)/100.0)+'\% </td></tr>'
    resHtml+='<tr><td>Precision :</td><td> '+str(int(precision*10000)/100.0)+'\% </td></tr>'
    resHtml+='<tr><td>Recall : </td><td>'+str(int(recall*10000)/100.0)+'\% </td></tr>'
    resHtml+='<tr><td> FM : </td><td>'+str(int(fm*10000)/100.0)+'\% </td></tr>'
    resHtml+='</table>\n<hr>\n'
    resHtml+='<table><tr><td>sample</td><td>Acc</td><td>Precision</td><td>Recall</td><td>FMeasure</td><tr>\n'
    for sampleFname in accDict.keys():
        fname=sampleFname.split('/')[-1]
        acc,pr,rec,fm=accDict[sampleFname]
        resHtml+='<tr><td><a href="'+fname+'">'+fname.split('.')[0]+'</a></td><td>'+str(int(10000*(acc))/100.0)+'%</td><td>'
        resHtml+=str(int(10000*(pr))/100.0)+'%</td><td>'+str(int(10000*(rec))/100.0)+'%</td><td>'+str(int(10000*(fm))/100.0)+'%</td></tr>\n'
    resHtml+='</table></body></html>'
    open(p['outReportDir']+'index.html','w').write(resHtml)

    
def printHelp(out=sys.stdout):
    name=sys.argv[0].split('/')[-1]
    helpStr=name+""" usage:
icdarCh4Task4:
    """+name+""" icdarCh4Task4 GT_SPRINTF_PATTERN file1 file2 ... fileN
    The pattern is a description of how to get the ground-truth name from a sample.
    eg:"""+name+"""  icdarCh4Task4 './data/ch4tsk4/gt/%s.txt' ./data/ch4tsk4/example1/*txt
    TODO: implement "dont care"
icdarCh2Task4:
    """+name+""" icdarCh2Task4 GT_SPRINTF_PATTERN file1 file2 ... fileN
    The pattern is a description of how to get the ground-truth name from a sample.
    eg:"""+name+""" ./pe.py  icdarCh2Task4 './data/ch2tsk4/gt/%s.txt' ./data/ch2tsk4/sampleMethod/*txt
    TODO: implement "dont care"
getFCNReport:
    """+name+""" getFCNReport outputDir GT_SPRINTF_PATTERN IMG_SPRINTF_PATTERN 
    file1 file2 ... fileN
    The program generates an HTML report for 2point rectangles the output 
    directory must not exist.
fsnsMetrics:
    """+name+""" fsnsMetrics gtFile submission1 submission2 ... submissionN
    Both submission and gt files contain dictionaries mapping sample-image
    filenames to transcriptions in json format.
    """
    out.write(helpStr)


if __name__=='__main__':
    #this serves as demonstration of how to use the the python routines defined
    #in the module
    if len(sys.argv)<2:
        printHelp()
        sys.exit(0)

    if sys.argv[1]=='icdarCh4Task4':
        gtFnameLambda = (lambda fname:(sys.argv[2]%(fname.split('/')[-1].split('.')[0])))
        gtSampleNameTuples=[(gtFnameLambda(f),f) for f in sys.argv[3:]]
        fm,pr,rec=get4pEndToEndMetric(gtSampleNameTuples)
        print 'Precision: %3.2f\nRecall   : %3.2f\nF-Measure: %3.2f'%(pr,rec,fm)
        sys.exit(0)

    if sys.argv[1]=='icdarCh2Task4':
        gtFnameLambda = (lambda fname:(sys.argv[2]%(fname.split('/')[-1].split('.')[0])))
        gtSampleNameTuples=[(gtFnameLambda(f),f) for f in sys.argv[3:]]
        fm,pr,rec=get2pEndToEndMetric(gtSampleNameTuples)
        print 'Precision: %3.2f\nRecall   : %3.2f\nF-Measure: %3.2f'%(pr,rec,fm)
        sys.exit(0)

    if sys.argv[1]=='getFCNReport':
        outDir=sys.argv[2]
        gtDir=sys.argv[3]
        submFiles=sys.argv[4:]
        imgGtSubmFnames=[(gtDir+f.split('/')[-1].split('.')[0]+'.jpg',gtDir+f.split('/')[-1].split('.')[0]+'.txt',f) for f in submFiles]
        getReport(imgGtSubmFnames,dontCare='###',outReportDir=outDir)
        sys.exit(0)

    if sys.argv[1]=='fsnsMetrics':
        gtFname=sys.argv[2]
        gtDict=json.loads(open(gtFname).read())
        print '|     Method | Precision |   Recall  |  F-Measure |  Seq. Err |'
        for methodName in sys.argv[3:]:
            submissionDict=json.loads(open(methodName).read())
            FM,precision,recall,seqErr=getFSNSMetrics(submissionDict,gtDict)[:4]
            name=methodName.split('/')[-1].split('.')[0]
            print '| %10s |     %5.2f |     %5.2f |      %5.2f |     %05.2f |'%(name,precision*100,recall*100,FM*100,seqErr*100)
            #ofname='.'.join(methodName.split('.')[:-1])+'.evaluation.json'
        sys.exit(0)
    print 'Unknown option '+sys.argv[1]+'  !Aborting!'
    printHelp()
    sys.exit(1)
