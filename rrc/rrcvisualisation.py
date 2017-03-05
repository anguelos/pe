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
import os
import time
from commands import getoutput as go #fast access to the shell


try:
    import cv2
    imwrite=cv2.imwrite
    imread=cv2.imread
    __have_cv2__=True
except:
    #if opencv is not available but PIL is it can alternatively be used for IO
    import PIL as pil
    imwrite=lambda filename,imgArray:pil.Image.fromarray(imgArray).save(filename)
    imread= lambda filename:numpy.array(pil.Image.open(filename))
    __have_cv2__=False

import rrcmetrics
import rrcio


def plotSample(img,gtFname,subFname):
    p={'fig':None,'gt'}


def get2pEndToEndVisualisationData(idSubmGt,**kwargs):
    p={'dontCare':'###','iouThr':.5,'maxEdist':0}
    p.update(kwargs)
    allRelevant=0
    allRetrieved=0
    correct=0
    sampleDict={}
    for sampleId in idSubmGt.keys():
        gtStr,submStr = idSubmGt[sampleId]
        #jsonFile='.'.join(imgPath.split('.')[:-1])+'.json'
        gtLoc,gtTrans=rrcio.loadBBoxTranscription(gtStr)
        submLoc,submTrans=rrcio.loadBBoxTranscription(submStr)
        if gtLoc.shape[1]==8:
            gtLoc=rrcio.conv4pointToLTBR(gtLoc)
        if submLoc.shape[1]==8:
            submLoc=rrcio.conv4pointToLTBR(submLoc)
        IoU=rrcmetrics.get2PointIoU(gtLoc,submLoc)[0]
        edDist=rrcmetrics.getEditDistanceMat(gtTrans,submTrans)[0]
        if p['dontCare']!='':
            IoU,edDist=rrcmetrics.filterDontCares(IoU,edDist,gtTrans,p['dontCare'])
        allRelevant+=IoU.shape[0]
        allRetrieved+=IoU.shape[1]
        d={}
        d['iou']=IoU.tolist()
        d['eddist']=edDist.tolist()
        d['subm']=[[submLoc[k].tolist(),submTrans[k]] for k in range(len(submTrans))]
        d['gt']=[[gtLoc[k].tolist(),gtTrans[k]] for k in range(len(gtTrans))]
        d['correct']=np.sum((IoU>=p['iouThr'])*(edDist<=p['maxEdist'])*rrcmetrics.maskNonMaximalIoU(IoU,1))
        sampleDict[sampleId]=d
        correct+=np.sum((IoU>=p['iouThr'])*(edDist<=p['maxEdist'])*rrcmetrics.maskNonMaximalIoU(IoU,1))
    precision=float(correct)/allRetrieved
    recall=float(correct)/allRelevant
    FM=(2*precision*recall)/(precision+recall+.00000000000001)
    if rrcmetrics.get2pEndToEndMetric(idSubmGt.values())!=(FM,precision,recall):#SANITY AGSAINST rrcmetric.py
        raise Exception('Visualisation Doesnt match respective metric function')
    return {'globals':{'fm':FM,'precision':precision,'recall':recall},'persample':sampleDict}



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


def getReport(imgFnameGtSubmFdata,**kwargs):
    #TODO FIX UGLY FIX
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
        raise Exception('Output directory must not exist')
    go('mkdir -p '+p['outReportDir'])
    startTime=time.time()
    for inImgFname,gtFname,submFname in imgFnameGtSubmFdata:
        sampleFname=p['outReportDir']+'/'+inImgFname.split('/')[-1].split('.')[0]
        rectSubm,transcrSubm=rrcio.loadBBoxTranscription(submFname)
        rectGt,transcrGt=rrcio.loadBBoxTranscription(gtFname)
        rectGt=rrcio.conv4pointToLTBR(rectGt)
        rectSubm=rrcio.conv4pointToLTBR(rectSubm)
        IoU=rrcmetrics.get2PointIoU(rectGt,rectSubm)[0]
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
            #resTbl+='</td><td>'.join([str(int(k*10000)/100.0) for k in IoU[k,:]*strEqual[k,:]])+'</td></tr>\n'
            resTbl+='</td><td>Exception thrown</td></tr>\n'#UGLY FIX
        resTbl+='</table>\n'
        resHtml='<html><body>\n<h3>'+inImgFname.split('/')[-1].split('.')[0]+'</h3>\n'
        acc=0
        try:
            acc=((IoU*strEqual).sum()/float(IoU.shape[1]))#UGLYFIX
        except:
            pass
        if p['dontCare']!='':
            precision=np.nan
            try:
                precision=(IoU*strEqual).max(axis=1)[(strCare*IoU).sum(axis=1)>0].mean()#UGLY FIX
            except:
                pass
            if np.isnan(precision):
                precision=0
            recall=np.nan
            try:
                recall=(IoU*strEqual).max(axis=0)[(strCare).sum(axis=0)>0].mean()#UGLY FIX
            except:
                pass
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
    gt
    for k in range(len(allIoU)):
        submSize,gtSize=allIoU[k].shape
        IoU[submIdx:submIdx+submSize,gtIdx:gtIdx+gtSize]=allIoU[k]
        strEqual[submIdx:submIdx+submSize,gtIdx:gtIdx+gtSize]=allEqual[k].T
        strCare[submIdx:submIdx+submSize,gtIdx:gtIdx+gtSize]=allCare[k].T
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
