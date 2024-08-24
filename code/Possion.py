import cv2
import numpy as np
from scipy.sparse import lil_matrix, linalg

def Poisson(dst, mask, lap):
    # 1. 计算求解的像素个数
    loc = np.nonzero(mask)
    num = loc[0].shape[0] 
    # 2. 有多少个像素个数则需要多少个方程，需要构造num*num大小的稀疏矩阵和num大小的b向量
    A = lil_matrix((num, num), dtype=np.float64)
    b = np.ndarray((num,), dtype=np.float64)
    # 3. 要将每个像素映射到0~num-1的索引之中，因为A系数矩阵也是根据索引求构造的
    hhash = {(x, y): i for i, (x, y) in enumerate(zip(loc[0], loc[1]))}
    # 用于找上下左右四个像素位置
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    height, width = dst.shape[:2]

    # 求出边界的位置
    boundary_mask = np.ones((dst.shape[0], dst.shape[1]))
    boundary_mask[np.ix_(np.arange(1, dst.shape[0] - 1), np.arange(1, dst.shape[1] - 1))] = 0
    _boundary = np.nonzero(boundary_mask)
    boundary = {(x, y): i for i, (x, y) in enumerate(zip(_boundary[0], _boundary[1]))}

    # 4. 构造A系数矩阵和b向量
    for i, (x, y) in enumerate(zip(loc[0], loc[1])):
        if (x, y) in boundary:
                A[i, i] = 1
                b[i]=dst[(x,y)]
                continue
        A[i, i] = -4
        b[i] = lap[x, y]
        p = [(x + dx[j], y + dy[j]) for j in range(4)]
        for j in range(4):
            if p[j] in hhash:
                A[i, hhash[p[j]]] = 1
            else:
                if 0 <= p[j][0] < height and 0 <= p[j][1] < width:
                    b[i] -= dst[p[j]]


    # 5. 由于A是稀疏矩阵，可以将lilmatrix转成cscmatrix，方便矩阵运算
    A = A.tocsc()
    # 6. 求解X
    X = linalg.splu(A).solve(b)
    
    # 7. 将X复制到对应位置
    result = np.copy(dst)
    for i, (x, y) in enumerate(zip(loc[0], loc[1])):
        if(0<x<height-1 and 0<y<width-1):
            X[i] = max(0, min(255, X[i]))
            result[x, y] = X[i]

    return result

def Seamless_cloning(src, dst, mask, flag):
    # 1. 计算mask区域的外切矩形
    loc = np.nonzero(mask)
    xbegin = max(np.min(loc[0]) - 1, 0)
    xend = max(np.max(loc[0]) + 1, 0)
    ybegin = max(np.min(loc[1]) - 1, 0)
    yend = max(np.max(loc[1]) + 1, 0)

    # 如果要做Monochrome transfer，求ROI前将原图像转为单色
    if flag == 2:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 2. 根据mask区域范围截取出对应的mask、src
    cutMask = mask[xbegin:xend, ybegin:yend]
    cutSrc = src[xbegin:xend, ybegin:yend]
    finalMask = np.zeros((dst.shape[0], dst.shape[1]))
    finalSrc = np.zeros_like(dst)
    finalLap = np.zeros_like(dst, dtype=np.float64)

    # 如果要做Monochrome transfer，src和lap都是单通道，只复制长宽
    if flag == 2:
        finalSrc = np.zeros((dst.shape[0], dst.shape[1]))
        finalLap = np.zeros((dst.shape[0], dst.shape[1]), dtype=np.float64)

    # 3. 复制到对应位置
    finalMask[xbegin:xend, ybegin:yend] = cutMask
    finalSrc[xbegin:xend, ybegin:yend] = cutSrc

    # 4. 求散度
    if flag == 0:  # seamless cloning
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        srcGrad = cv2.filter2D(np.float64(finalSrc), -1, kernel)
        finalLap = srcGrad
    elif flag == 1:  # mixed seamless cloning
        kernel = [np.array([[0, -1, 1]]), np.array([[1, -1, 0]]),
                  np.array([[0], [-1], [1]]), np.array([[1], [-1], [0]])]
        grads = [np.where(np.abs(cv2.filter2D(np.float64(finalSrc), -1, k)) > np.abs(cv2.filter2D(np.float64(dst), -1, k)),
                         cv2.filter2D(np.float64(finalSrc), -1, k),
                         cv2.filter2D(np.float64(dst), -1, k)) for k in kernel]
        finalLap = np.sum(grads, axis=0)
    elif flag == 2:  # Monochrome transfer
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        srcGrad = cv2.filter2D(np.float64(finalSrc), -1, kernel)
        srcGrad = np.repeat(srcGrad[:, :, np.newaxis], 3, axis=2)
        finalLap = srcGrad

    # 5. 逐通道求解
    result = [Poisson(a, finalMask, b) for a, b in zip(cv2.split(dst), cv2.split(finalLap))]

    # 6. 合并三个通道
    final = cv2.merge(result)
    return final

def Texture_flattening(src, mask, threshold1, threshold2):
    # 1. 求图像边缘
    edges = cv2.Canny(src, threshold1, threshold2)

    # 2. 用于计算src四个方向梯度的核
    kernel = [np.array([[0, -1, 1]]), np.array([[1, -1, 0]]),
              np.array([[0], [-1], [1]]), np.array([[1], [-1], [0]])]

    # 用于检测四个方向边缘点的核
    kernelEdge = [np.array([[0, 1, 1]]), np.array([[1, 1, 0]]),
                  np.array([[0], [1], [1]]), np.array([[1], [1], [0]])]

    # 3. 检测四个方向边缘点
    edges = [cv2.filter2D(edges, -1, kernelEdge[i]) for i in range(4)]

    # 4. 计算src四个方向梯度
    grads = [cv2.filter2D(np.float64(src), -1, k) for k in kernel]

    # 不是边缘点的将梯度修改为0
    for i in range(4):
        grads[i][edges[i] == 0] = 0

    # 5. 计算src的散度
    finalLap = np.sum(grads, axis=0)
    
    # 6. 逐通道求解
    result = [Poisson(a, mask, b) for a, b in zip(cv2.split(src), cv2.split(finalLap))]

    # 7. 合并三个通道
    final = cv2.merge(result)
    return final

def Local_illumination_changes(src, mask, a, b):
    # 1. 对图像进行对数变换（避免log(0)的问题，加1）
    log_src = np.log(np.float64(src) + 1)
    
    # 2. 计算对数变换后图像的散度
    laplacian = cv2.Laplacian(log_src, -1, ksize=1)
    
    # 3. 计算mask区域的外切矩形
    loc = np.nonzero(mask)
    xbegin, xend, ybegin, yend = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1])
    
    # 4. 根据mask区域范围截取出对应的mask
    cutMask = mask[xbegin:xend, ybegin:yend]
    
    # 5. 为了方便后面计算复制的位置，将mask变成和src一样大
    finalMask = np.zeros((src.shape[0], src.shape[1]))  
    finalMask[xbegin:xend, ybegin:yend] = cutMask

    # 6. 求绝对值
    normal_value = np.linalg.norm(laplacian)
    
    # 7. 套公式求引导场
    finalLap = (a ** b) * ((normal_value) ** (-b)) * laplacian

    # 8. 逐通道求解
    final_result = []
    for dst, lap in zip(cv2.split(src), cv2.split(finalLap)):
        log_dst = np.log(dst + 1)
        # 1. 计算求解的像素个数
        loc = np.nonzero(finalMask)
        num = loc[0].shape[0]
        # 2. 构造num*num大小的稀疏矩阵A和num大小的b向量
        A = lil_matrix((num, num), dtype=np.float64)
        b = np.ndarray((num,), dtype=np.float64)
        # 3. 将每个像素映射到0~num-1的索引之中
        hhash = {(x, y): i for i, (x, y) in enumerate(zip(loc[0], loc[1]))}
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        height, width = dst.shape[:2]

        # 4. 构造A系数矩阵和b向量
        for i, (x, y) in enumerate(zip(loc[0], loc[1])):
            A[i, i] = -4
            b[i] = lap[x, y]
            p = [(x + dx[j], y + dy[j]) for j in range(4)]
            for j in range(4):
                if p[j] in hhash:
                    A[i, hhash[p[j]]] = 1
                else:
                    if 0 <= p[j][0] < height and 0 <= p[j][1] < width:
                        b[i] -= log_dst[p[j]]

        # 5. 由于A是稀疏矩阵，可以将lilmatrix转成cscmatrix，方便矩阵运算
        A = A.tocsc()
        # 6. 求解X
        X = linalg.splu(A).solve(b)

        # 7. 将X复制到对应位置
        result = np.copy(dst)
        for i, (x, y) in enumerate(zip(loc[0], loc[1])):
            X[i] = max(0, min(255, X[i]))
            result[x, y] = np.exp(X[i]) - 1
        final_result.append(result)

    # 9. 合并三个通道
    final = cv2.merge(final_result)
    return final

def Local_color_changes(src, mask, R, G, B):
    # 1. 分离三个颜色通道
    b_channel, g_channel, r_channel = cv2.split(src)
    
    # 2. 调整每个通道的强度
    r_channel = np.clip(r_channel * R, 0, 255).astype(np.uint8)
    g_channel = np.clip(g_channel * G, 0, 255).astype(np.uint8)
    b_channel = np.clip(b_channel * B, 0, 255).astype(np.uint8)
    
    # 3. 将各通道合并为新图像
    modified_src = cv2.merge((b_channel, g_channel, r_channel))

    # 4. 计算mask区域的外切矩形
    loc = np.nonzero(mask)
    xbegin, xend, ybegin, yend = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1])
    
    # 5. 根据mask区域范围截取出对应的mask
    cutMask = mask[xbegin:xend, ybegin:yend]
    
    # 6. 将mask变成和src一样大
    finalMask = np.zeros((src.shape[0], src.shape[1]))
    finalMask[xbegin:xend, ybegin:yend] = cutMask

    # 7. 计算Laplacian
    laplacian = cv2.Laplacian(np.float64(modified_src), -1, ksize=1)

    # 8. 逐通道求解
    result = [Poisson(a, finalMask, b) for a, b in zip(cv2.split(src), cv2.split(laplacian))]

    # 9. 合并三个通道
    final = cv2.merge(result)
    return final

def Seamless_tiling(src):
    # 1. 融合的mask是整个图像矩形区域
    mask = np.ones((src.shape[0], src.shape[1]))

    # 2. 求出边界的位置
    boundary_mask = np.ones((src.shape[0], src.shape[1]))
    boundary_mask[np.ix_(np.arange(1, src.shape[0] - 1), np.arange(1, src.shape[1] - 1))] = 0
    _boundary = np.nonzero(boundary_mask)
    boundary = {(x, y): i for i, (x, y) in enumerate(zip(_boundary[0], _boundary[1]))}
    
    loc = np.nonzero(mask)
    num = loc[0].shape[0]
    A = lil_matrix((num, num), dtype=np.float64)
    hhash = {(x, y): i for i, (x, y) in enumerate(zip(loc[0], loc[1]))}
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    # 3. 逐通道求解A矩阵和b向量
    result = []
    for _src in cv2.split(src):
        # 3.1 求解原图散度
        _lap = cv2.Laplacian(np.float64(_src), -1, ksize=1)
        
        # 3.2 计算边界新的像素值
        tmp = np.zeros_like(_src)
        tmp[0] = (_src[0] + _src[-1]) * 0.5
        tmp[-1] = (_src[0] + _src[-1]) * 0.5
        tmp[:, 0] = (_src[:, 0] + _src[:, -1]) * 0.5
        tmp[:, -1] = (_src[:, 0] + _src[:, -1]) * 0.5
        b = np.ndarray((num,), dtype=np.float64)

        # 3.3 遍历图像每个像素
        for i, (x, y) in enumerate(zip(loc[0], loc[1])):
            # 如果像素是边界点，则b[i]为融合的边界值，A[i,i]=1
            if (x, y) in boundary:
                b[i] = tmp[x, y]
                A[i, i] = 1
            # 如果像素不是边界点，则b[i]为散度值，A[i,i] = -4，A[i,邻居] = 1
            else:
                b[i] = _lap[x, y]
                A[i, i] = -4
                p = [(x + dx[j], y + dy[j]) for j in range(4)]
                for j in range(4):
                    A[i, hhash[p[j]]] = 1

        A = A.tocsr()
        # 3.4 求解像素值
        X = linalg.spsolve(A, b)
        
        # 3.5 将X复制到对应位置
        res = np.zeros_like(_src)
        for i, (x, y) in enumerate(zip(loc[0], loc[1])):
            X[i] = max(0, min(255, X[i]))
            res[x, y] = X[i]
        result.append(res)

    # 4. 合并三个通道
    final = cv2.merge(result)
    
    # 5. 返回拼接后的结果
    result_img = np.tile(final, (2, 3, 1))
    return result_img

