###RRC

This code contains a python implemetation of performance evaluation metrics for text localisation and recognition evaluation.

#### Command line example

* *Robust reading competition end-to-end evaluation for axis oriented rectangles:*
```bash
python ./examples/sampleCli.py  icdarCh2Task4 './examples/data/ch2tsk4/gt/%s.txt' ./examples/data/ch2tsk4/sampleMethod/img_*txt
```

* *Robust reading competition end-to-end evaluation of quadrilaterals:*
```bash
python ./rrc.py  icdarCh2Task4 './examples/data/ch4tsk4/gt/%s.txt' ./examples/data/ch4tsk4/example1/*txt
```

* *Robust reading competition all text in image evaluation:*
```bash
python ./rrc.py fsnsMetrics ./examples/data/fsns/gt.json ./examples/data/fsns/fake*
```

* *Robust reading competition all text in image evaluation:*
This code for this functionality has still many bugs. But overall it is a demonstration of how to use the code effectivelly.
```bash
rm -Rf /tmp/output;clear;python ./rrc.py  getFCNReport /tmp/output/  './data/fcnReport/%s.gt.txt' './data/fcnReport/%s.jpg'  ./data/fcnReport/img_*.res.txt
firefox /tmp/output/index.html
```

#### Web service example

* *Start server:*
```bash
python ./examples/sampleWeb.py
```
* *Client from the CLI:*
```bash
curl -F "gtFile=@./data/ch2tsk4/gt.tar.gz" -F "submissionFile=@./data/ch2tsk4/sampleMethod.tar.gz" -F "format=json"   http://127.0.0.1:8080/evaluate
```

* *Web Client:*
```bash
firefox http://127.0.0.1:8080
```
upload a pair of archive files found in ./examples/data/ch...
