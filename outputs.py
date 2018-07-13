def output1(inputImg):
    finalArr1 = np.empty(shape=(len(inputImg), len(inputImg[0])), dtype='object')
    finalArr1[:] = ' '
    for value in range(1, newLabel):
        flag = False
        print('--------------------',value)
        arr = np.empty(shape=(len(inputImg), len(inputImg[0])), dtype='object')
        arr[:] = ' '
        for j in range(len(labeling)):
            for i in range(len(labeling[0])):
                if (labeling[j][i] == value):
                    flag = True
                    arr[j][i] = str(inputImg[j][i])

        if flag and (arr != ' ').sum() > Background:
            print(arr)
            for j in range(len(labeling)):
                for i in range(len(labeling[0])):
                    if (labeling[j][i] == value):
                        finalArr1[j][i] = str(arr[j][i])

    data_visualization_2dr(np.array(finalArr1), 'Connected Components', visualize=True)

def output2(inputImg):
    finalArr1 = np.empty(shape=(len(inputImg), len(inputImg[0])), dtype='object')
    finalArr1[:] = ' '
    for value in range(1, newLabel):
        flag = False
        print('--------------------',value)
        arr = np.empty(shape=(len(inputImg), len(inputImg[0])), dtype='object')
        arr[:] = ' '
        for j in range(len(labeling)):
            for i in range(len(labeling[0])):
                if (labeling[j][i] == value):
                    flag = True
                    arr[j][i] = str(value)

        if flag and (arr != ' ').sum() > Background:
            print(arr)
            for j in range(len(labeling)):
                for i in range(len(labeling[0])):
                    if (labeling[j][i] == value):
                        finalArr1[j][i] = str(arr[j][i])

    data_visualization_2dr(np.array(finalArr1), 'Connected Components', visualize=True)