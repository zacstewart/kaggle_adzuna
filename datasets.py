import csv

def loadFile(path='data/Train_rev1.csv'):
  f = open(path)
  reader = csv.reader(f)
  data = [row for row in reader]
  header = data[0]
  header_map = dict(zip(header, range(len(header))))
  rows = data[1:]

  return (header_map, rows)
