from glob import glob
import os
import subprocess
from time import sleep

inps = glob('*.inp')

for inp in inps:
  subprocess.call('qsub -j y -l h_rt=01:00:00 /usr/local/bin/abaqus611job job=%s interactive'%(inp), shell=True)

output = 'abaqus'
while 'abaqus' in output:
  sleep(2)
  p = subprocess.Popen('Qstat',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  output, errors = p.communicate()
  print 'Waiting...'

print 'done'
f=open('DONEFILE.txt', 'w')
f.write('DONE!')
f.close()
