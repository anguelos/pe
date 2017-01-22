Command line invocation:

*Robust reading competition end-to-end evaluation for axis oriented rectangles:
```bash
python ./rrc.py  icdarCh2Task4 './data/ch2tsk4/gt/%s.txt' ./data/ch2tsk4/sampleMethod/img_*txt
```

*Robust reading competition end-to-end evaluation of quadrilaterals:
```bash
python ./rrc.py  icdarCh2Task4 './data/ch4tsk4/gt/%s.txt' ./data/ch4tsk4/example1/*txt
```

*Robust reading competition all text in image evaluation:
```bash
python ./rrc.py fsnsMetrics ./data/fsns/gt.json ./data/fsns/fake*
```

*Robust reading competition all text in image evaluation:
```bash
rm -Rf /tmp/output;clear;python ./rrc.py  getFCNReport /tmp/output/  './data/fcnReport/%s.gt.txt' './data/fcnReport/%s.jpg'  ./data/fcnReport/img_*.res.txt
firefox /tmp/output/index.html
```
