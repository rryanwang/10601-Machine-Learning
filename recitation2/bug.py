import ipdb
ipdb.set_trace()

#reverse the rows of a 2D array
def reverse(original):
    rows = len(original)
    cols = len(original[0])
    new = [[0]*cols]*rows
    for i in range(rows):
        for j in range(cols):
            oppositeRow = rows-i
            new[oppositeRow][j]=original[i][j]
    return new
a = [[1,2],
    [3,4],
    [5,6]]
print(reverse(a))