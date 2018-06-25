"""
dicom切割类
"""
import os
import vtk
import time
import argparse
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np
import skimage.io as io
import shutil
import pydicom


def addPadding(number):
    newnumber = '0000'+number
    length = len(newnumber)
    return newnumber[length-4:length]


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


# 灰度三通道变rgb三通道
def grey_to_red(train):
    rgb = np.zeros((train.shape[0], train.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = train*1.5
    return rgb


# 切割dicom
def FilterDicom(dicomPath, stlfilepath, savepath, orderdir):
    new_path = savepath
    # 读取dicom文件
    dicomImageData, m_Origin = ReadDicomFile(dicomPath)
    # 读取stl
    stlmapImageData = processStlToImageData(dicomImageData, m_Origin, stlfilepath)
    # 处理文件夹
    checkAndMakeDir(new_path)
    # 获取stl和dicom的array数据
    count, dicomArray, stlmapArray = getResharpedAryFromImageData(dicomImageData, stlmapImageData)
    # count等于0代表读取image文件存在问题
    if count == 0:
        print("folder " + orderdir + " image size error!")
        return
    dicomshape = dicomArray.shape
    dicomfiles = os.listdir(dicomPath)
    # 操作numpy数据
    for i in range(dicomshape[0]):
        train = stlmapArray[i, :, :]
        # 上下镜像翻转
        train = train[::-1]
        train = np.where(train > 135, 1, 0)
        if np.any(train > 0):
            ori = dicomArray[i, :, :]
            ori = ori[::-1]
            dicom_full_path = os.path.join(dicomPath, dicomfiles[i])
            ds = pydicom.dcmread(dicom_full_path)
            # 还原回无符号的正整数
            new = (ori-ds.RescaleIntercept) / ds.RescaleSlope
            # 与label相乘
            ori = ori * train.astype(np.uint16)
            new = new * train.astype(np.uint16)
            new = new.astype(np.uint16)
            ds.PixelData = new.tostring()
            pydicom.dcmwrite(dicom_full_path.replace(dicomPath, savepath), ds)
            io.imsave(new_path + orderdir + '-' + addPadding(str(i)) + '.png', winchange(ori, 0, 1000))
    print("folder " + orderdir + " done!")


def getResharpedAryFromImageData(dicomImageData, stlmapImageData):
    # 保存label
    # 转化标量位一维数组
    stlmapArray = vtk_to_numpy(stlmapImageData.GetPointData().GetScalars())
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


def processStlToImageData(dicomImageData, m_Origin, stlfilepath):
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

    # 获取处理的数据
    return flip2.GetOutput()


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


def checkAndMakeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def getCurrentTime():
    return str(time.strftime("%a %b %d %Y %H:%M:%S ", time.localtime()))+':'


def _process(args, orderDir):
    # 消除warning
    import warnings

    warnings.filterwarnings("ignore")
    path = args.root_dir + "/" + orderDir
    if os.path.isdir(path):
        print(getCurrentTime() + "switch work dir to path:" + path)
        dicomdir = os.path.join(path, args.dicom_path)
        stlpath = os.path.join(path, args.stl_path)

        # 判断dicom文件夹是否存在
        if not os.path.exists(dicomdir):
            print("%s order:%s escape process, DICOM dir can't find!" % (getCurrentTime(), orderDir))
            return
        # 检验stl是否存在
        realStlPath = path + '/' + args.part + '.stl'
        if not os.path.exists(stlpath):
            print("%s order:%s escape process, stl can't find! " % (getCurrentTime(), orderDir))
            return

        if os.path.exists(realStlPath):
            os.remove(realStlPath)
            print('remove unkown stl at path:' + realStlPath)
        print('copying stl file to order dir path!')
        shutil.copy(stlpath, realStlPath)
        partFolder = checkSaveDir(args, orderDir)
        try:
            FilterDicom(dicomdir, realStlPath, partFolder, orderDir)
        finally:
            if os.path.exists(realStlPath):
                os.remove(realStlPath)
            print('remove stl file at:' + realStlPath)


def checkSaveDir(args, orderDir):
    checkAndMakeDir(args.save_dir)
    orderFolder = os.path.join(args.save_dir, orderDir)
    checkAndMakeDir(orderFolder)
    partFolder = os.path.join(orderFolder, args.part)
    checkAndMakeDir(partFolder)
    return partFolder


if __name__ == "__main__":
    # 读取命令行参数
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--mode', type=str, default="s")
    parser.add_argument('--part', type=str, default=None)
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--dicom-path', type=str, default=None)
    parser.add_argument('--stl-path', type=str, default=None)
    # parser.add_argument('--parallel', type=str, default='f')
    args = parser.parse_args()
    # 多线程模式
    # print('parallelmode:' + args.parallel)
    # 部位
    print('part:' + args.part)
    # 操作根目录
    print('rootdir:' + args.root_dir)
    # 保存数据的目录，需要已经创建
    print('savedir:' + args.save_dir)
    # dicom文件的子路径，例如DICOM/A
    print('dicom-path:' + args.dicom_path)
    # stl文件的子路径，例如 STL/肝脏.stl
    print('stl-path:' + args.stl_path)

    dirlist = os.listdir(args.root_dir)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=3) as executor:
        for orderDir in dirlist:
            executor.submit(_process, args, orderDir)




