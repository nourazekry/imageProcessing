import math
from PIL import Image
import csv
import numpy as np 
import warnings
import sys


# gather input
menu = input("How would you like to edit your image?\n a.Crop\n b.Flip\n c.Scale\n d.Rotate\n e.Linear Mapping\n f.Power-Mapping\n g.Histogram\n h.Convolution\n i.Edge Detection\n j.Min Filtering\n k.Median Filtering\n l.Max Filtering\n m.Object Tracking Prototype\n")


if (menu != "m"):
    chosenImage = input("Choose Image to edit: ")
    img = Image.open(chosenImage).convert("L")
    img_array = np.array(img)

def convolution(img_array, new_arr, kernel):
    img_array = [[1], [1], [1]]
    m = len(kernel)
    n = len(kernel[0])
    # M = int(img_array.shape[0])
    # N = int(img_array.shape[1])
    M = 1
    N = 3

    # calculate index
    mx = int((m - 1)/2)
    nx = int((n - 1)/2)
    for x in range(M):
        for y in range(N):
            # for every pixel in image
            res = 0
            for i in range(m):
                for j in range(n):
                    # for every element in kernel
                    l = x + mx - i
                    k = y + nx - j
                    # for every element in neighborhood
                    if (l < 0 or l >= M or k < 0 or k >= N):
                        # add zero because zero padding
                        res+= 0
                    else:
                        # add to sum
                        res+= img_array[l,k] * kernel[i][j]
            new_arr[x,y] = res

def blurConvolution(img_array, new_arr):
    kernel = [[2/159,4/159,5/159,4/159,2/159],
    [4/159,9/159,12/159,9/159,4/159],
    [5/159,12/159,15/159,12/159,5/159],
    [4/159,9/159,12/159,9/159,4/159],
    [2/159,4/159,5/159,4/159,2/159]]
    convolution(img_array, new_arr, kernel)

def reverseSobel(difference, new_arr, kernelA, kernelB, t):
# blur difference
    blurred = np.zeros(difference.shape, dtype=np.uint8)
    blurConvolution(difference, blurred)
    warnings.filterwarnings('ignore')

# M'' = blurred - complement of blurred + 255
    bin = np.zeros(difference.shape, dtype=np.uint8)
    complement = np.invert(blurred)
    for x in range(M):
        for y in range(N):
            i = blurred[x,y]
            j = complement[x,y]
            bin[x,y] = i - j + 255
# abs (sobelV * M'') + abs(sobelH * M'') i.e produce gradient image
    sobelGradientImage(bin, new_arr, kernelA, kernelB, t)
# blur gradient image
    

def sobelGradientImage(img_array, new_arr, kernelA, kernelB, t):
    # identical to convolution but with two kernels and gradient magnitude
    m = len(kernelA)
    n = len(kernelA[0])
    M = int(img_array.shape[0])
    N = int(img_array.shape[1])
    mx = int((m - 1)/2)
    nx = int((n - 1)/2)
    for x in range(M):
        for y in range(N):
            resA = 0
            resB = 0
            for i in range(m):
                for j in range(n):
                    l = x + mx - i
                    k = y + nx - j

                    if (l < 0 or l >= M or k < 0 or k >= N):
                        resA+= 0
                        resB+=0
                    else:
                        # convolve with both  
                        resA+= img_array[l,k] * kernelA[i][j]
                        resB+= img_array[l,k] * kernelB[i][j]
            # calculate gradient magnitude
            G = abs(resA) + abs(resB)
            # make binary edge map using threshold
            if G > t:
                new_arr[x,y] = 255
            else:
                new_arr[x,y] = 0
def bilinearInterpolation (x, y, x0, y0):
    # this is (x0,y0)
    x0 = math.floor(x0)
    y0 = math.floor(y0)
    # this is (x0 + 1, y0 + 1)
    x01 = math.ceil(x0)
    y01 = math.ceil(y0)
    # calculate difference
    s = x - x0
    t = y - y0

    # derive fBar(x,y)
    xY0 = (1-s)*(img_array[x0, y0]) + s * img_array[x01, y0]
    xY01 = (1 - s) * img_array[x0, y01] + s * img_array[x01, y01]
    xy = (1 - t) * xY0 + t * xY01

    return xy

if menu == "a":
    print("The image's current dimensions are ", img_array.shape[0], "x", img_array.shape[1], ". To crop: ")
    row1 = input("Enter new start row: ")
    row2 = input("Enter new end row: ")
    col1 = input("Enter new start column: ")
    col2 = input("Enter new end column: ")
    # truncate array using given indices
    img_array = img_array[int(row1)+1 :int(row2)+1, int(col1)+1:int(col2)+1]
    processed_img = Image.fromarray(img_array)
    processed_img.save(r"edited.jpeg")

elif menu == "b":
    dim = input("Would you like to flip the image (a) horizontally or (b) vertically? ")

    if dim == "a":
        start = 0
        end = img_array.shape[1]
        row = -1
        # flip over x-axis
        for x in img_array:
            row += 1
            col = -1
            for y in x:
                col += 1
                # stop at the middle because everything will be reflected
                if(col == end/2):
                    break
                # switch values at reflection point
                newCol = (end - 1) - col
                temp = img_array[row,newCol]
                img_array[row, newCol] = img_array[row, col]
                img_array[row, col] = temp
        processed_img = Image.fromarray(img_array)
        processed_img.save(r"edited.jpeg")

    elif dim == "b":
        start = 0
        end = img_array.shape[0]
        row = -1
        # reflect over y-axis
        for x in img_array:
            row += 1
            # check midpoint
            if(row == end/2):
                break
            col = -1
            for y in x:
                col += 1
                newRow = (end - 1) - row
                temp = img_array[newRow,col]
                img_array[newRow, col] = img_array[row, col]
                img_array[row, col] = temp
        processed_img = Image.fromarray(img_array)
        processed_img.save(r"edited.jpeg")

elif menu == "c":
    # assign dimensions and get user input
    M = int(img_array.shape[0])
    N = int(img_array.shape[1])
    print("The image's current dimensions are ", M, "x", N)
    P = int(input("Please enter new vertical dimension: "))
    Q = int(input("Please enter new horizontal dimension: "))
    new_arr = np.zeros((P, Q, 3), dtype=np.uint8)

    for x in range(P):
        for y in range(Q):
            # figure out the index at which i will modify new array
            u = x*((M-1)/(P-1))
            v = y*((N-1)/(Q-1))
            # apply bilinear interpolation
            new_arr[x,y] = bilinearInterpolation(x, y, u, v)
    processed_img = Image.fromarray(new_arr)
    processed_img.save(r"edited.jpeg")

elif menu == "d":
    # assign dimensions
    M = int(img_array.shape[0])
    N = int(img_array.shape[1])
    a = float(input("Please enter angle of rotation: "))
    # calculate new origin point
    P = abs(round((M * math.cos(a)) + (N * math.sin(a))))
    Q = abs(round((N * math.cos(a)) + (M * math.sin(a))))
    new_arr = np.zeros((P, Q, 3), dtype=np.uint8)
    originOldX = M / 2
    originOldY = N / 2

    originNewX = P/2
    originNewY = Q/2

    for x in range(P):
        for y in range(Q):
            # calculate new x and y values
            newX = ((x-originNewX) * math.cos(a)) + ((y - originNewY) * math.sin(a))
            newY = -((x-originNewX) * math.sin(a)) + ((y - originNewY) * math.cos(a))
            newX = round(originOldX + newX)
            newY = round(originOldY + newY)
            # only assign new array if it is within original image, else zero padding
            if (newX in range(M) and newY in range(N)):
                new_arr[x, y] = img_array[newX,newY]
    processed_img = Image.fromarray(new_arr)
    processed_img.save(r"edited.jpeg")
elif menu == "e":
    M = int(img_array.shape[0])
    N = int(img_array.shape[1])
    new_arr = np.zeros((M, N, 3), dtype=np.uint8)
    # get input parameters
    a = float(input("Enter parameter a, contrast: "))
    b = float(input("Enter parameter b, brightness: "))
    for x in range(M):
        for y in range(N):
            # apply to image
            new_arr[x,y] = a * img_array[x, y] + b
    processed_img = Image.fromarray(new_arr)
    processed_img.save(r"edited.jpeg")
elif menu == "f":
    M = int(img_array.shape[0])
    N = int(img_array.shape[1])
    # max graylevels
    L = 256
    new_arr = np.zeros((M, N, 3), dtype=np.uint8)
    # get input
    gamma = float(input("Enter gamma to be used for power-law mapping {(L−1)[u/(L −1)]^gamma}: "))

    # apply power-law mapping to image
    for x in range(M):
        for y in range(N):
            new_arr[x,y] = (L-1) * (img_array[x, y]/ (L-1)) ** gamma
    processed_img = Image.fromarray(new_arr)
    processed_img.save(r"edited.jpeg")

elif menu == "g":
    M = int(img_array.shape[0])
    N = int(img_array.shape[1])
    L = 256
    hist = np.zeros((L,), dtype=int)
    histc =  np.zeros((L,), dtype=int)
    histcn = np.zeros((L,), dtype=float)
    new_arr = np.zeros(img_array.shape, dtype =  np.uint8)
    original_stdout = sys.stdout

    for x in range(M):
        for y in range(N):
            hist[img_array[x,y]] += 1

    # print histogram to file
    with open('histogram.txt', 'w') as f:
        sys.stdout = f
        for i in range(L):
            print(i, ": ", hist[i])
        sys.stdout = original_stdout

    # perform histogram equalization
    equalize = input("equalize? (y/n): ")
    if equalize == "y":
        for i in range(L):
            for j in range(i + 1):
                # calculate cumulative and normalized cumulative histograms
                histc[i] = hist[j] + histc[i]
                histcn[i] = histc[i]/(M*N)
       
        # equalize histogram
        eqhist = np.floor((L-1) * histcn).astype(np.uint8)
        # make image 1d
        imgFlat = list(img_array.flatten())
        # remap to image
        eqFlat = [eqhist[i] for i in imgFlat]
        new_arr = np.reshape(np.asarray(eqFlat), img_array.shape)


        processed_img = Image.fromarray(new_arr)
        processed_img.save(r"edited.jpeg")
    
elif menu == "h":

    M = int(img_array.shape[0])
    N = int(img_array.shape[1])

    new_arr = np.zeros((M, N, 3), dtype=np.uint8)
    new_arr = np.zeros((1, 3,), dtype=np.uint8)

    # read in kernel
    file = input("CSV File to read kernel from: ")
    kernel = []
    flag = 0
    # parse kernel
    while flag != 1:
        try:
            with open(file, 'r') as fd:
                reader = csv.reader(fd)
                for line in reader:
                    row = []
                    for s in line:
                        if '/' in s:
                            n, d = s.split('/')
                            s = (float(n)/float(d))
                        row.append(float(s))
                    kernel.append(row)  
            flag = 1   
        except FileNotFoundError:
            file = input("CSV File to read kernel from: ")
            flag = 0
    # apply convolution using kernel array
    convolution(img_array, new_arr, kernel)
    print(new_arr)
    # processed_img = Image.fromarray(new_arr)
    # processed_img.save(r"edited.jpeg")
elif menu == "i":
    M = int(img_array.shape[0])
    N = int(img_array.shape[1])

    sobel_v_arr = np.zeros((M, N, 3), dtype=np.uint8)
    sobel_h_arr = np.zeros((M, N, 3), dtype=np.uint8)
    final_arr = np.zeros((M, N, 3), dtype=np.uint8)

    sobelV = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobelH = [[-1,-2,-1], [0,0,0], [1, 2, 1]]

    t = int(input("Enter edge detection threshold >= 0 and <= 255: "))

    # apply sobel horizontal and vertical then get gradient image
    
    sobelGradientImage(img_array, final_arr, sobelV, sobelH, t)
    processed_img = Image.fromarray(final_arr)
    processed_img.save(r"edited.jpeg")

elif menu == "j":
    M = int(img_array.shape[0])
    N = int(img_array.shape[1])

    new_arr = np.zeros(img_array.shape, dtype=np.uint8)
    size = input("Enter size M for a kernel of size MxM: ")
    size = int(size)
    mx = int((size - 1)/2)
    nx = int((size - 1)/2)
    for x in range(M):
        for y in range(N):
            res = []
            for i in range(size):
                for j in range(size):
                    l = x + mx - i
                    k = y + nx - j
                    if (l < 0 or l >= M or k < 0 or k >= N):
                        res.append(0)
                    else:
                        res.append(img_array[l,k])
            res.sort()
            new_arr[x,y] = res[0]
    processed_img = Image.fromarray(new_arr)
    processed_img.save(r"edited.jpeg")
elif menu == "k":
    M = int(img_array.shape[0])
    N = int(img_array.shape[1])

    new_arr = np.zeros(img_array.shape, dtype=np.uint8)
    size = input("Input size M for kernel of size MxM: ")
    size = int(size)
    mx = int((size - 1)/2)
    nx = int((size - 1)/2)
    for x in range(M):
        for y in range(N):
            res = []
            for i in range(size):
                for j in range(size):
                    l = x + mx - i
                    k = y + nx - j
                    if (l < 0 or l >= M or k < 0 or k >= N):
                        res.append(0)
                    else:
                        res.append(img_array[l,k])
            res.sort()
            new_arr[x,y] = res[int(len(res)/2) -1]
    processed_img = Image.fromarray(new_arr)
    processed_img.save(r"edited.jpeg")
elif menu == "l":
    M = int(img_array.shape[0])
    N = int(img_array.shape[1])

    new_arr = np.zeros(img_array.shape, dtype=np.uint8)
    size = input("Input size M for kernel of size MxM: ")
    size = int(size)
    mx = int((size - 1)/2)
    nx = int((size - 1)/2)
    for x in range(M):
        for y in range(N):
            res = []
            for i in range(size):
                for j in range(size):
                    l = x + mx - i
                    k = y + nx - j
                    if (l < 0 or l >= M or k < 0 or k >= N):
                        res.append(0)
                    else:
                        res.append(img_array[l,k])
            res.sort()
            new_arr[x,y] = res[len(res)-1]
    processed_img = Image.fromarray(new_arr)
    processed_img.save(r"edited.jpeg")
elif menu == "m":
    # images to be used
    # video frames
    img_real = Image.open(r"video/000.jpg").convert("L")
    img_subsequent_real = Image.open(r"video/099.jpg").convert("L")

    img_array_real_1 = np.array(img_real)
    img_array_real_2 = np.array(img_subsequent_real)


    # image array shape
    M = int(img_array_real_1.shape[0])
    N = int(img_array_real_1.shape[1])
    
    # new arrays for video frame edges
    real_final_1 = np.zeros(img_array_real_1.shape, dtype=np.uint8)
    real_final_2 = np.zeros(img_array_real_1.shape, dtype=np.uint8)

    # new array for edge difference 
    diff = np.zeros(img_array_real_1.shape, dtype=np.uint8)

    final_arr = np.zeros(img_array_real_1.shape, dtype=np.uint8)

    # sobel convolution kernels
    sobelV = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobelH = [[-1,-2,-1], [0,0,0], [1, 2, 1]]

    # edge detection by sobel for video frames
    sobelGradientImage(img_array_real_1, real_final_1, sobelV, sobelH, 240)
    sobelGradientImage(img_array_real_2, real_final_2, sobelV, sobelH, 240)
    # subtract video frame edges
    diff = real_final_2 - real_final_1

    # apply uncanny filter with threshold 240
    reverseSobel(diff, final_arr, sobelV, sobelH, 240)

    processed_img = Image.fromarray(final_arr)
    processed_img.save(r"ReverseSobel.jpeg")

