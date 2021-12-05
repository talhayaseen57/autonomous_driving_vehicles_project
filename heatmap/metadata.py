def count(meta_data:list):
    """takes list as an input [[frame_number, element_type, is_good]]"""
    count = {}
    for data in meta_data:
        try:
            count[data[1][0]]+=1
        except:
            count[data[1]]=[1,0]
        if data[2]: count[data[1][1]]+=1
    for element in count:
        element.append(element[1]/element[0])
    return count
