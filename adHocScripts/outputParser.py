import re

re1 = re.compile(r'cost (\d.\d+)')
re2 = re.compile(r'f=(\d.\d+)')
re3 = re.compile(r'cr=(\d.\d+)')

costs = []
f = []
cr = []

infile = open('DE_1398912077/test.sh.o8380916', 'r')
for l in infile:
    costs.extend(re1.findall(l))
    f.extend(re2.findall(l))
    cr.extend(re3.findall(l))
infile.close()
    
outputs = zip(costs, f, cr)
outfile = open('de_outs.csv', 'w')
for i, o in enumerate(outputs):
    c = ', '.join(o)
    outfile.write('%s,%s\n'%(i, c))
outfile.close()
    



