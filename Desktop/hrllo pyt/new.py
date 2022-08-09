a="234648"

b=list(a)
b.sort()
print(b)
d=len(b)

c=a.index(b[d-2])
d=a.index(b[d-1])
length=d-c
height=b[d-2]
height=int(height)
g=length*height
print(length*height)