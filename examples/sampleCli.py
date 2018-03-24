#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from rrc import *

def printHelp(out=sys.stdout):
    name=sys.argv[0].split('/')[-1]
    helpStr=name+""" usage:
icdarCh4Task4:
    """+name+""" icdarCh4Task4 GT_SPRINTF_PATTERN file1 file2 ... fileN
    The pattern is a description of how to get the ground-truth name from a sample.
    eg:"""+name+"""  icdarCh4Task4 './examples/data/ch4tsk4/gt/%s.txt' ./examples/data/ch4tsk4/example1/*txt
    
icdarCh2Task4:
    """+name+""" icdarCh2Task4 GT_SPRINTF_PATTERN file1 file2 ... fileN
    The pattern is a description of how to get the ground-truth name from a sample.
    eg:"""+name+"""  icdarCh2Task4 './examples/data/ch2tsk4/gt/%s.txt' ./examples/data/ch2tsk4/sampleMethod/*txt
    
getFCNReport:
    """+name+""" getFCNReport outputDir GT_SPRINTF_PATTERN IMG_SPRINTF_PATTERN 
    file1 file2 ... fileN
    The program generates an HTML report for 2point rectangles the output 
    directory must not exist.

fsnsMetrics:
    """+name+""" fsnsMetrics gtFile submission1 submission2 ... submissionN
    Both submission and gt files contain dictionaries mapping sample-image
    filenames to transcriptions in json format.

mAP:
    """+name+""" mAP embedding_tsv_file.
    The tsv will have onw row per entity, the first column will be a a number or string representing the class and all
    other columns are the embedding. 
    """
    out.write(helpStr)


if __name__=='__main__':
    #this serves as demonstration of how to use the the python routines defined
    #in the module
    if len(sys.argv)<2:
        printHelp()
        sys.exit(0)

    if sys.argv[1].lower()=='map':
        lines = [l.split("\t") for l in  open(sys.argv[2]).read().strip().split("\n") if l[0]!='#']
        labels = np.array([l[0] for l in lines],dtype=str)
        embeddings = np.array([[float(col) for col in l[1:]] for l in lines])
        mAP,AP=map.get_map(labels=labels, embeddings=embeddings)
        print 'mAP: %5.3f\n'%(mAP)
        sys.exit(0)

    if sys.argv[1]=='icdarCh2Task4':
        gtFnameLambda = (lambda fname:(sys.argv[2]%(fname.split('/')[-1].split('.')[0])))
        gtSampleDataTuples=[(open(gtFnameLambda(f)).read(),open(f).read()) for f in sys.argv[3:]]
        fm,pr,rec=get2pEndToEndMetric(gtSampleDataTuples,dontCare='###')
        print 'Precision: %3.2f\nRecall   : %3.2f\nF-Measure: %3.2f'%(pr,rec,fm)
        sys.exit(0)

    if sys.argv[1]=='getFCNReport':
        outDir=sys.argv[2]
        gtPat=sys.argv[3]
        imgPat=sys.argv[4]
        submFiles=sys.argv[5:]
        imgGtSubmFnames=[((imgPat%(f.split('/')[-1].split('.')[0])),(gtPat%(f.split('/')[-1].split('.')[0])),f) for f in submFiles]
        getReport(imgGtSubmFnames,dontCare='###',outReportDir=outDir)
        sys.exit(0)

    if sys.argv[1]=='fsnsMetrics':
        gtFname=sys.argv[2]
        gtDict=json.loads(open(gtFname).read())
        print '|     Method | Precision |   Recall  |  F-Measure |  Seq. Err |  Combined |'
        for methodName in sys.argv[3:]:
            submissionDict=json.loads(open(methodName).read())
            combinedSoftMetric,FM,precision,recall,seqErr=getFSNSMetrics(submissionDict,gtDict)[:5]
            name=methodName.split('/')[-1].split('.')[0]
            print '| %10s |     %5.2f |     %5.2f |      %5.2f |     %05.2f |     %05.2f |'%(name,precision*100,recall*100,FM*100,seqErr*100,combinedSoftMetric*100)
            #ofname='.'.join(methodName.split('.')[:-1])+'.evaluation.json'
        sys.exit(0)
    print 'Unknown option '+sys.argv[1]+'  !Aborting!'
    printHelp()
    sys.exit(1)
