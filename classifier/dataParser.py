def parser(path):
    """takes path for a txt file as string,
    and returns the parsed data"""

    lines = []
    with open(path) as f:
        lines = f.readlines()

    data = []

    for line in lines:
        record = line.split()
        data.append([record[0],int(record[1])])

    return data