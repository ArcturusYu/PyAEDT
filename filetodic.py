def file_to_dict17(file_name):
    """
    Reads a dictionary from a text file. Assumes each line in the file contains one key-value pair, separated by a comma.
    If a key is duplicated in the file, the values are aggregated into a list.
    
    Parameters:
        file_name (str): The path to the file from which to read the dictionary.
    
    Returns:
        dict: The dictionary read from the file, with all values of the same key merged into a list.
    """
    dict_data = {}
    with open(file_name, 'r') as file:
        for line in file:
            key, value = line.strip().split(',', 1)
            if key in dict_data:
                # If the key already exists in the dictionary, append the new value to the list of values for this key.
                if not isinstance(dict_data[key], list):
                    # Convert existing value to a list if it's not already a list
                    dict_data[key] = [dict_data[key]]
                dict_data[key].append(value)
            else:
                # Create a new key in the dictionary with the value in a list
                dict_data[key] = [value]
    return dict_data


# def file_to_dict(filename):
#     import re
#     data_dict = {}
    
#     # Open and read the file
#     with open(filename, 'r') as file:
#         data = file.read().splitlines()
        
#     # Process every 181 rows as a single block
#     for i in range(0, len(data), 181):
#         # Extract the key (first row of block) and convert from string tuple to actual tuple of floats
#         key_str = data[i].split(',')[0].strip()
#         key = tuple(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", key_str)))
        
#         # The remaining rows form the list of complex numbers
#         values_str = ''.join(data[i+1:i+181])
#         # Match all complex numbers in the format (a+bj)
#         values = [complex(v) for v in re.findall(r'[\d\.\+\-e]+(?:[\+\-][\d\.\+\-e]+j)', values_str)]
        
#         data_dict[key] = values
    
#     return data_dict

def file_to_dict(filename):
    import re
    data_dict = {}

    # Open and read the file
    with open(filename, 'r') as file:
        data = file.read().splitlines()

    # Process every 181 rows after the first as values
    for i in range(0, len(data), 181):
        # Extract the key from the first part of the line, before the comma that follows the parenthesis
        line = data[i]
        key_part, first_value = re.split(r'\),\[', line)
        key_part += ')'
        
        # Convert the key string to an actual tuple of floats
        key = tuple(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", key_part)))
        
        values_str = first_value+''.join(data[i+1:i+181])
        # Match all complex numbers in the format (a+bj)
        values = [complex(v) for v in re.findall(r'[\s\-]\d.\d+e[\+\-]\d+[\+\-]\d.\d+e[\+\-]\d+j', values_str)]
        data_dict[key] = values

    return data_dict



rEPhiDic = file_to_dict('D:\pythontxtfile\eEPhi_example.txt')
All17PositionDic = file_to_dict17('D:\pythontxtfile\All17Position.txt')
# print(All17PositionDic['1'])