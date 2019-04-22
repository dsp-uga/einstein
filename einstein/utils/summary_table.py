def draw(year, target_hr, grid, model_name, dictionary):
    '''Draws a summary table representing the regression statistics

    Arguments:
        year (str):
            Year NAM-NMM data belongs to
        target_hr (int):
            Target hour offset the predictions belong to
        grid (str):
            Grid option selected by user, which represents the grid around
            ATHENS region
        model_name (str):
            Model that the data has been trained on
        dictionary (dict):
            Dictionary of metric names and values
    '''
    print("-" * 75)
    print(f'{" " * 25} Statistics Summary Table')
    print("-" * 75)
    print(f'{" " * 15}NAM-NMM {year} Data - ({grid}, {grid}) Grid Size')

    if target_hr == '1':
        print(f'Target Hour Offset {" " * 12} : {" " * 20} {target_hr} hr')
    else:
        print(f'Target Hour Offset {" " * 12} : {" " * 20} {target_hr} hrs')

    print(f'Model {" " * 25} : {" " * 10} {model_name}')

    for key, value in dictionary.items():
        key_length = len(str(key))
        if key_length < 30:
            print(f'{key} {" " * (30 - key_length)} : {" " * 20} {value}')
        else:
            print(f'{key} {" " * 5} : {" " * 5} {value}')

    print("-" * 75)
