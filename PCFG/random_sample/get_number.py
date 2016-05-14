

if __name__ == "__main__":
	input = "combine.txt"
	
	total = 0
	recurse = 0
	i = 1
	with open(input) as myfile:
		for line in myfile:
			data = line.split()
			num = int(data[1])
			if i % 2 == 0:
				# recurse
				recurse += num
			else:
				total += num			
			i += 1
	print "total:", total
	print "recurse:", recurse
	print "percent:", (recurse + 0.0)/total
	
