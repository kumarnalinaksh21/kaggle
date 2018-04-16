import csv
def load(fn,isTrain=True):
    data = []
    with open(fn, 'rb') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i==0: continue
            if isTrain: mapped_ = [int(row[0]), int(row[1]), int(row[2]), row[3], row[4], int(row[5])]
            else: mapped_ = [int(row[0]), row[1], row[2]]
            data.append(mapped_)
    return data 
            

