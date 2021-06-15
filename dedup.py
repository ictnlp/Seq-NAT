f=open("out.de",encoding='utf-8').readlines()
o=open("out.de.dedup","w",encoding='utf-8')
for line in f:
    words=line.split(' ')
    l = len(words)
    s=''
    for i in range(l):
        if i==0 or words[i]!=words[i-1]:
            s = s+words[i]
            if i!=l-1:
                s = s+' '
    o.write(s)
