# coding: utf-8
import os
import hashlib
from glob import glob

flist = []
fmd5 = []
dl = []

dirname = './data/custom/train/15_hour'

for e in ['png', 'jpg']: flist.extend(glob('%s/*.%s'%(dirname,e)))

for fn in flist:
  with open(fn, 'rb') as fin:
    data = fin.read()
    m = hashlib.md5(data)
    fmd5.append(m.hexdigest())

for i in range(len(flist)):
  if flist[i] in dl: continue
  for j in range(i+1, len(flist)):
    if flist[j] in dl: continue
    if fmd5[i] == fmd5[j] and not flist[j] in dl:
      dl.append(flist[j])

for a in dl: os.remove(a)