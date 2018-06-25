"""
拼合多个stl的label制作，单线程版！
"""
import os
import vtk
import time
import argparse
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np
import skimage.io as io
from PIL import Image
import shutil
from concurrent.futures import ProcessPoolExecutor

# 融合两张图片
def blend_two_images(ori, label, savepath):
    rgb = np.zeros((ori.shape[0], ori.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = ori
    rgb[:, :, 1] = ori
    rgb[:, :, 2] = ori
    img1 = Image.fromarray(rgb, "RGB")
    img1 = img1.convert('RGBA')

    img2 = Image.fromarray(label, "RGB")
    img2 = img2.convert('RGBA')

    img = Image.blend(img1, img2, 0.3)
    img.save(savepath)


# 灰度三通道变rgb三通道
def grey_to_red(train):
    rgb = np.zeros((train.shape[0], train.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = train*1.5
    return rgb


# 变换窗宽窗位
def winchange(pix, wl, ww):

    window_center = wl  # 窗位
    window_width = ww  # 窗宽
    win_min = (2 * window_center - window_width) / 2.0 + 0.5
    win_max = (2 * window_center + window_width) / 2.0 + 0.5

    # 计算pix为ct值
    ct = pix # * slope + intercept

    # 计算比例后乘以255的单位矩阵
    ct = ((ct - window_center) / (win_max - win_min) + 0.5) * 255

    # 消除大于最大ct值的数值，设为窗口最大值
    ct = np.where(ct > 255, 255, ct)

    # 消除小于最小ct值的数值，设为窗口最小值
    ct = np.where(ct < 0, 0, ct)

    ct = ct.astype(np.uint8)
    return ct


# 制作Label数据
def MakeLabel(dicomPath, stlfilepath, savepath, orderdir, wl, ww):
    MakeLabel(dicomPath, stlfilepath, savepath, orderdir, wl, ww, 'S')


# 制作Label数据
def MakeLabel(dicomPath, stlfilepaths, savepath, orderdir, wl, ww, face):
    # 定义保存路径
    label_path, ori_path, train_path = getSavepath(savepath)
    # 读取dicom文件
    dicomImageData, m_Origin = ReadDicomFile(dicomPath)
    # 读取stl
    stlmapImageDatas = processStlToImageData(dicomImageData, m_Origin, stlfilepaths)
    # 处理文件夹
    checkAndMakeDir(label_path)
    checkAndMakeDir(ori_path)
    checkAndMakeDir(train_path)
    # 获取stl和dicom的array数据
    count, dicomArray, stlmapArray = getResharpedAryFromImageData(dicomImageData, stlmapImageDatas)
    # count等于0代表读取image文件存在问题
    if count == 0:
        print("folder " + orderdir + " image size error!")
        return
    dicomshape = dicomArray.shape
    # 横断面
    if face == 'S':
        # 操作numpy数据
        for i in range(dicomshape[0]):
            if i % 2 > 0:
                train = stlmapArray[i, :, :]
                # 上下镜像翻转
                train = train[::-1]
                train = np.where(train > 135, 135, 0)
                io.imsave(train_path + orderdir + '-' + addPadding(str(i)) + '.png', train)
                ori = dicomArray[i, :, :]
                ori = ori[::-1]
                ori = winchange(ori, wl, ww)
                io.imsave(ori_path + orderdir + '-' + addPadding(str(i)) + '.jpg', ori)
                blend_two_images(ori, grey_to_red(train), label_path + orderdir + '-' + addPadding(str(i)) + '.png')
        print("folder " + orderdir + " done!")
        print("train data has save to path: " + train_path)
        print("original data has save to path: " + ori_path)
        print("label data has save to path: " + label_path)
    # 冠状面
    if face == 'A':
        # 操作numpy数据
        for i in range(dicomshape[1]):
            if i % 2 > 0:
                train = stlmapArray[:, i, :]
                train = np.where(train > 135, 135, 0)
                io.imsave(train_path + orderdir + '-' + addPadding(str(i)) + '.png', train)
                ori = dicomArray[:, i, :]
                ori = winchange(ori, wl, ww)
                io.imsave(ori_path + orderdir + '-' + addPadding(str(i)) + '.jpg', ori)
                blend_two_images(ori, grey_to_red(train), label_path + orderdir + '-' + addPadding(str(i)) + '.png')
        print("folder " + orderdir + " done!")
        print("train data has save to path: " + train_path)
        print("original data has save to path: " + ori_path)
        print("label data has save to path: " + label_path)
    # 矢状面
    if face == 'R':
        # 操作numpy数据
        for i in range(dicomshape[2]):
            if i % 2 > 0:
                train = stlmapArray[:, :, i]
                train = np.where(train > 135, 135, 0)
                io.imsave(train_path + orderdir + '-' + addPadding(str(i)) + '.png', train)
                ori = dicomArray[:, :, i]
                ori = winchange(ori, wl, ww)
                io.imsave(ori_path + orderdir + '-' + addPadding(str(i)) + '.jpg', ori)
                blend_two_images(ori, grey_to_red(train), label_path + orderdir + '-' + addPadding(str(i)) + '.png')
        print("folder " + orderdir + " done!")
        print("train data has save to path: " + train_path)
        print("original data has save to path: " + ori_path)
        print("label data has save to path: " + label_path)


def getResharpedAryFromImageData(dicomImageData, stlmapImageDatas):
    stlmapArray = None
    # 保存label
    # 转化标量位一维数组
    for stlmapImageData in stlmapImageDatas:
        if stlmapArray is None:
            stlmapArray = vtk_to_numpy(stlmapImageData.GetPointData().GetScalars())
        else:
            stlmapArray = stlmapArray + vtk_to_numpy(stlmapImageData.GetPointData().GetScalars())

    dicomArray = vtk_to_numpy(dicomImageData.GetPointData().GetScalars())
    if stlmapArray.size % (512*512) > 0:
        print('image size error!')
        return 0, None, None
    try:
        # 转置一维数组为三维数据，首维为图片数量
        count = int(stlmapArray.size / (512 * 512))
        stlmapArray = stlmapArray.reshape((count, 512, 512))
        dicomArray = dicomArray.reshape((count, 512, 512))

        return count, dicomArray, stlmapArray
    except Exception :
        print(Exception)
        return 0, None, None


def processStlToImageData(dicomImageData, m_Origin, stlfilepaths):
    resultary=None
    for stlfilepath in stlfilepaths:
        if not os.path.exists(stlfilepath):
            continue
        polyData = readStlFile(stlfilepath)

        orginlData = dicomImageData
        spacing = orginlData.GetSpacing()
        outval = 0
        whiteData = vtk.vtkImageData()
        whiteData.DeepCopy(orginlData)

        pointdata = whiteData.GetPointData()
        pointscalars = pointdata.GetScalars()

        # 通过矩阵计算将whiteData中点的颜色全部设置成白色
        sc = vtk_to_numpy(pointscalars)
        sc = np.where(sc < 255, 255, 255)
        newscalars = numpy_to_vtk(sc)
        pointdata.SetScalars(newscalars)
        whiteData.Modified()

        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(polyData)
        pol2stenc.SetOutputOrigin(m_Origin)
        pol2stenc.SetOutputSpacing(spacing)
        pol2stenc.SetOutputWholeExtent(orginlData.GetExtent())
        pol2stenc.Update()

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(whiteData)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()
        flip = vtk.vtkImageFlip()
        flip.SetInputData(imgstenc.GetOutput())
        flip.SetFilteredAxes(1)
        flip.Update()
        flip2 = vtk.vtkImageFlip()

        flip2.SetInputData(flip.GetOutput())
        flip2.SetFilteredAxes(2)
        flip2.Update()
        if resultary is None:
            resultary = [flip2.GetOutput()]
        else:
            resultary.append(flip2.GetOutput())
    return resultary


def processDICOMToImageX(dicomImageData, origin, extent):
    center =[0,0,0]
    centers = [0,0,0]
    center[0] = origin[0] + spacing[0] * 0.5 * (extent[0] + extent[1])
    center[1] = origin[1] + spacing[1] * 0.5 * (extent[2] + extent[3])
    center[2] = origin[2] + spacing[2] * 0.5 * (extent[4] + extent[5])
    centers[0] = center[0]
    centers[1] = center[1]
    centers[2] = center[2]
    imagecast = vtk.vtkImageCast()
    imagecast.SetInputConnection(dicomImageData)
    imagecast.SetOutputScalarTypeToChar()
    imagecast.ClampOverflowOn()
    imagecast.Update()
    imagecast.SetUpdateExtentToWholeExtent()
    pImageResliceY = vtk.vtkImageReslice()
    pImageResliceY.SetInputConnection(imagecast.GetOutputPort())
    pImageResliceY.SetOutputDimensionality(2)
    pImageResliceY.SetResliceAxesDirectionCosines(coronalX, coronalY, coronalZ)
    pImageResliceY.SetResliceAxesOrigin(center)
    pImageResliceY.SetInterpolationModeToLinear()
    pImageResliceY.Update()
    pImageResliceY.GetOutput()


def readStlFile(stlfilepath):
    stlreader = vtk.vtkSTLReader()
    stlreader.SetFileName(stlfilepath)
    stlreader.Update()
    # 处理stl
    polyData = stlreader.GetOutput()
    return polyData


def ReadDicomFile(dicomPath):
    # 读取dicom
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dicomPath)
    reader.Update()
    # intercept =reader.GetRescaleOffset()
    # slope = reader.GetRescaleSlope()
    dicomImageData = reader.GetOutput()
    m_Origin = [reader.GetImagePositionPatient()[0],
                reader.GetImagePositionPatient()[1],
                reader.GetImagePositionPatient()[2]]
    return dicomImageData, m_Origin


def getSavepath(savepath):
    train_path = savepath + '/train/'
    ori_path = savepath + '/ori/'
    label_path = savepath + '/label/'
    return label_path, ori_path, train_path


def checkAndMakeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def addPadding(number):
    newnumber = '0000'+number
    length = len(newnumber)
    return newnumber[length-4:length]


def getCurrentTime():
    return str(time.strftime("%a %b %d %Y %H:%M:%S ", time.localtime()))+':'


def _process(path, orderDir, args):
    # 消除warning
    import warnings
    warnings.filterwarnings("ignore")
    if os.path.isdir(path):
        print(getCurrentTime() + "switch work dir to path:" + path)
        dicomdir = path + '/' + args.dicom_path
        stlpaths = []
        realStlPaths = []
        i = 0
        for stl in args.stl_path.split(','):
            stlpaths.append(path + '/' + stl)
            realStlPaths.append(path + '/' + os.path.dirname(stl) + '/a' + str(i) + 'a.stl')
            i = i + 1
        if not os.path.exists(dicomdir):
            print(getCurrentTime() + "DICOM dir can't find!")
            return
        for j in range(len(stlpaths)):
            if os.path.exists(realStlPaths[j]):
                os.remove(realStlPaths[j])
                print('remove unkown stl at path:' + realStlPaths[j])
            if not os.path.exists(stlpaths[j]):
                continue
            print('copying stl file to order dir path!')
            shutil.copy(stlpaths[j], realStlPaths[j])

        savedir = args.save_dir + '/'
        checkAndMakeDir(savedir)

        orderFolder = savedir + orderDir + '/'
        checkAndMakeDir(orderFolder)

        partFolder = orderFolder + args.part + '/'
        checkAndMakeDir(partFolder)
        try:
            MakeLabel(dicomdir, realStlPaths, partFolder, orderDir, args.wl, args.ww, args.face)
        finally:
            for j in range(len(stlpaths)):
                if os.path.exists(realStlPaths[j]):
                    os.remove(realStlPaths[j])
                    print('remove stl file at:' + realStlPaths[j])


if __name__ == "__main__":
    # 消除warning
    import warnings

    warnings.filterwarnings("ignore")

    # 读取命令行参数
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--mode', type=str, default="s")
    parser.add_argument('--part', type=str, default=None)
    parser.add_argument('--face', type=str, default='S')
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--dicom-path', type=str, default=None)
    parser.add_argument('--stl-path', type=str, default=None)
    parser.add_argument('--wl', type=int, default=0)
    parser.add_argument('--ww', type=int, default=200)
    # parser.add_argument('--parallel', type=str, default='f')
    args = parser.parse_args()

    # 运行模式s为单文件夹模式，m为里面文件夹下所有子文件夹的多文件夹模式
    print('mode:' + args.mode)
    # 多线程模式
    # print('parallelmode:' + args.parallel)
    # 部位
    print('part:' + args.part)
    # 部位
    print('face:' + args.face)
    # 操作根目录
    print('rootdir:' + args.root_dir)
    # 保存数据的目录，需要已经创建
    print('savedir:' + args.save_dir)
    # dicom文件的子路径，例如DICOM/A
    print('dicom-path:' + args.dicom_path)
    # stl文件的子路径，例如 STL/肝脏.stl
    print('stl-path:' + args.stl_path)
    # 窗位 默认为0
    print('wl:' + str(args.wl))
    # 窗宽 默认200
    print('ww:' + str(args.ww))

    # executor = ProcessPoolExecutor(max_workers=3)
    dirlist = os.listdir(args.root_dir)
    for orderDir in dirlist:
        root = args.root_dir
        spath = args.root_dir+"/"+orderDir
        _process(spath, orderDir, args)



