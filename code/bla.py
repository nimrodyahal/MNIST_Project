def bound(a):
	rows=len(a)
	cols=len(a[0])
	minx=cols
	miny=rows
	maxx=-1
	maxy=-1
	for y in range(rows):
		for x in range(cols):
			if a[y][x] > 0.01:
				if minx > x: minx = x
				if maxx < x: maxx = x
				if miny > y: miny = y
				if maxy < y: maxy = y
	return minx, miny, maxx, maxy