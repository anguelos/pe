#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from bottle import route, run, request
import tarfile
from StringIO import StringIO
import sys;sys.path.append('./')
import rrc
import time


"""
requirements:
pip install --user bottle
"""
#https://bottlepy.org/docs/dev/tutorial.html#quickstart-hello-world



@route('/')
def greet():
    return """<form action="/evaluate" method="post" enctype="multipart/form-data">
  Output:<select name="format">
      <option value="json">json</option>
      <option value="html">html</option>
      </select>
  Give The Groundtruth: <input type="file" name="gtFile" />
  Give The Submission: <input type="file" name="submissionFile" />
  <input type="submit" value="Evaluate" />
</form>"""

@route('/evaluate', method='POST')
def getMetric():
    gtRequest=request.files.get('gtFile')
    submRequest=request.files.get('submissionFile')
    loadT=time.time()
    gt = rrc.loadArchiveAsFiledataDict(gtRequest.filename,gtRequest.file.read())
    subm = rrc.loadArchiveAsFiledataDict(submRequest.filename,submRequest.file.read())
    gtSubmData=rrc.packFiledataDicts(gt,subm)
    metricT=time.time()
    FM,precision,recall=rrc.get2pEndToEndMetric(gtSubmData)
    endT=time.time()
    if request.forms.get('format')=='json':
        return {'FM':FM,'Precision':precision,'Recall':recall}
    else:
        res= '<table><tr><td>Precisison: %f</td><td>Recall: %f</td><td>F-Measure: %f</td></tr></table>\n\n'%(100*precision,100*recall,100*FM)
        res+='<hr> Read files in in %d msec.\n'%(int(1000*(metricT-loadT)))
        res+='<hr> Computed metrics in %d msec.\n'%(int(1000*(endT-metricT)))
        res+='<hr> Total in %d msec.\n'%(int(1000*(time.time()-loadT)))
        return res



if __name__=='__main__':
    print 'Command line client:\ncurl -F "gtFile=@./data/ch2tsk4/gt.tar.gz" -F "submissionFile=@./data/ch2tsk4/sampleMethod.tar.gz" -F "format=json"   http://127.0.0.1:8080/evaluate'
    print '\nGUI client:\nfirefox http://127.0.0.1:8080\n\nServer running:'
    run(host='0.0.0.0', port=8080, debug=True)