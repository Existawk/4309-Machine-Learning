#Parker Skinner
#1001541467
#4309-001
def file_stats(pathname):
    array = []
    file = open(pathname,"r")
    for line in file:
        array.append(float(line))
    file.close()

    avg = sum(array)/len(array)
    stdsum = 0
    for i in array:
        stdsum+=(i-avg)**2
    stdev = ((stdsum)/(len(array)-1))**(1/2)
    return avg, stdev