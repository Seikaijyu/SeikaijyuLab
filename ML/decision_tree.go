package ml

import (
	"math"
)

// Node 表示决策树的一个节点
type Node struct {
	featureIndex int     // 用于划分数据的特征索引
	threshold    float64 // 用于划分数据的阈值
	left         *Node   // 左子节点
	right        *Node   // 右子节点
	value        int     // 叶节点的值
}

// DecisionTree 表示一个决策树
type DecisionTree struct {
	maxDepth        int   // 树的最大深度
	minSamplesSplit int   // 节点分裂所需的最小样本数
	root            *Node // 树的根节点
}

// splitResult 表示一个划分结果，包含了信息增益、阈值、特征索引、左子集和右子集
type splitResult struct {
	infoGain     float64 // 信息增益，用于衡量划分的优劣
	threshold    float64 // 用于划分数据的阈值
	featureIndex int     // 用于划分数据的特征索引
	leftSets     dataset // 左子集
	rightSets    dataset // 右子集
}

// dataset 表示一个数据集，包含了特征和标签
type dataset struct {
	X [][]float64 // 特征
	y []int       // 标签
}

// NewDecisionTree 创建一个决策树的新实例
//
//	决策树是一种监督学习算法，用于分类和回归任务
//	它基于特征的值来做决策，每个节点都是一个特征，每个分支都是特征的值
//	决策树的目标是创建一个模型，可以用于预测目标变量的值
//	这个类实现的决策树算法叫做 ID3 算法，它是一种贪心算法，每次选择信息增益最大的特征来划分数据
//	信息增益是用于衡量划分的优劣，它是根据熵的变化来计算的
//	熵是用于衡量数据的不确定性，熵越高，数据越不稳定，信息增益越大
//	信息增益越大，说明划分的效果越好
//	除此之外还有其他决策树算法，比如 C4.5 算法和 CART 算法
//	C4.5算法使用信息增益比来选择特征，但是它和ID3算法不同的是，它可以处理连续特征，CART 算法使用基尼不纯度来选择特征，它可以处理多分类任务
//	C4.5算法的优点是可以处理连续特征，CART算法的优点是可以处理多分类任务
//	C4.5算法的缺点是计算复杂度高，CART算法的缺点是不支持概率输出
//
//	maxDepth 表示树的最大深度，用于防止过拟合
//	minSamplesSplit 表示节点分裂所需的最小样本数
func NewDecisionTree(maxDepth, minSamplesSplit int) *DecisionTree {
	// 创建一个决策树实例
	return &DecisionTree{maxDepth: maxDepth, minSamplesSplit: minSamplesSplit}
}

// Fit 使用训练数据构建决策树
//
// X 是特征数据，格式为
// [
//
//	[0.1, 0.2, 0.3],
//	[0.4, 0.5, 0.6],
//	...
//
// ]
//
// y 是标签数据，格式为
// [0, 1, 0, 1, 0, ...]
//
// 其中 X 和 y 的长度应该相等
func (dt *DecisionTree) Fit(X [][]float64, y []int) {
	// 递归构建决策树
	dt.root = dt.buildTree(X, y, 0)
}

// buildTree 递归构建决策树
//
//	具体是通过递归的方式来构建决策树
//	如果样本数小于 minSamplesSplit 或者树的深度大于 maxDepth，则返回叶节点
//	找到最佳的特征和阈值来分割数据
//	如果信息增益大于0，则继续递归构建树
//	如果信息增益小于等于0，则返回叶节点
//	最后返回树的根节点
func (dt *DecisionTree) buildTree(X [][]float64, y []int, depth int) *Node {
	// 如果样本数小于 minSamplesSplit 或者树的深度大于 maxDepth，则返回叶节点
	numSamples := len(X)
	// 递归终止条件，如果样本数小于 minSamplesSplit 或者树的深度大于 maxDepth，则返回叶节点
	if numSamples < dt.minSamplesSplit || depth >= dt.maxDepth {
		// 返回叶节点
		return &Node{value: majorityVote(y), featureIndex: -1, threshold: -1, left: nil, right: nil}
	}
	// 找到最佳的特征和阈值来分割数据
	bestSplit := dt.getBestSplit(X, y)
	// 如果信息增益大于0，则继续递归构建树
	if bestSplit.infoGain > 0 {
		// 递归构建左子树和右子树
		leftSubtree := dt.buildTree(bestSplit.leftSets.X, bestSplit.leftSets.y, depth+1)
		rightSubtree := dt.buildTree(bestSplit.rightSets.X, bestSplit.rightSets.y, depth+1)
		// 返回一个内部节点
		return &Node{
			featureIndex: bestSplit.featureIndex, // 特征索引
			threshold:    bestSplit.threshold,    // 阈值
			left:         leftSubtree,            // 左子节点
			right:        rightSubtree,           // 右子节点
			value:        -1,                     // 表示这不是一个叶节点
		}
	}
	// 如果信息增益小于等于0，则返回叶节点
	return &Node{
		value:        majorityVote(y), // 返回叶节点
		featureIndex: -1,              // 特征索引
		threshold:    -1,              // 阈值
		left:         nil,             // 左子节点
		right:        nil,             // 右子节点
	}
}

// getBestSplit 找到最佳的特征和阈值来分割数据
//
//	具体是通过遍历每个特征，找到最佳的划分结果
//	然后返回最佳的划分结果
func (dt *DecisionTree) getBestSplit(X [][]float64, y []int) splitResult {
	// 初始化最佳划分结果
	bestSplit := splitResult{infoGain: math.Inf(-1)}
	// 获取特征的数量
	numFeatures := len(X[0])
	// 遍历每个特征，找到最佳的划分结果
	for featureIndex := 0; featureIndex < numFeatures; featureIndex++ {
		// 获取遍历的特征的值，得到的是一个切片，用于划分数据
		featureValues := getColumn(X, featureIndex)
		// 获取特征的唯一值
		thresholds := unique(featureValues)
		// 遍历特征的唯一值，找到最佳的划分结果
		for _, threshold := range thresholds {
			// 划分数据集，把数据集划分为左子集和右子集
			leftIndices, rightIndices := splitDataset(X, featureIndex, threshold)
			// 如果左子集和右子集的长度大于0，则计算信息增益
			if len(leftIndices) > 0 && len(rightIndices) > 0 {
				// 根据左子集和右子集的索引，划分数据集，然后计算信息增益
				infoGain := informationGain(y, leftIndices, rightIndices)
				// 如果信息增益大于最佳划分结果的信息增益，则更新最佳划分结果
				if infoGain > bestSplit.infoGain {
					// 更新最佳划分结果
					bestSplit = splitResult{
						infoGain:     infoGain,                                                        // 信息增益
						featureIndex: featureIndex,                                                    // 特征索引
						threshold:    threshold,                                                       // 阈值
						leftSets:     dataset{X: subset(X, leftIndices), y: subset(y, leftIndices)},   // 左子集
						rightSets:    dataset{X: subset(X, rightIndices), y: subset(y, rightIndices)}, // 右子集
					}
				}
			}
		}
	}
	return bestSplit
}

// informationGain 计算信息增益
//
// 信息增益是用于衡量划分的优劣，它是根据熵的变化来计算的
// 熵是用于衡量数据的不确定性，熵越高，数据越不稳定
// 信息增益越大，说明划分的效果越好
// 信息增益的计算公式为
//
// 信息增益 = 总体熵 - (左子集熵 * 左子集权重 + 右子集熵 * 右子集权重)
//
// 其中总体熵为
//
// 总体熵 = - Σ (p * log2(p))
//
// 其中左子集熵和右子集熵分别为
//
// 左子集熵 = - Σ (p * log2(p))
// 右子集熵 = - Σ (p * log2(p))
//
// 其中 p 为每个类别的概率
//
// y 是标签数据
// left 和 right 分别是左子集和右子集的索引
//
// 返回信息增益
func informationGain(y, left, right []int) float64 {
	var totalEntropy, leftEntropy, rightEntropy float64
	// 计算信息增益
	totalEntropy = entropy(y)
	// 计算左子集和右子集的熵
	leftEntropy = entropy(subset(y, left))
	rightEntropy = entropy(subset(y, right))
	// 计算信息增益，具体来说使用了加权平均，使用左子集和右子集的权重除以总体数量得到的是左子集和右子集的权重
	weightL := float64(len(left)) / float64(len(y))
	weightR := float64(len(right)) / float64(len(y))
	// 信息增益 = 总体熵 - (左子集熵 * 左子集权重 + 右子集熵 * 右子集权重)
	return totalEntropy - (weightL*leftEntropy + weightR*rightEntropy)
}

// entropy 计算熵
//
//	熵是用于衡量数据的不确定性，熵越高，数据越不稳定
//	熵的计算公式为
//
//	熵 = - Σ (p * log2(p))
//
//	其中 p 为每个类别的概率
//
//	y 是标签数据
//
//	返回熵
func entropy(y []int) float64 {
	// 首先得到标签的总数
	total := float64(len(y))
	// 如果没有标签，则返回0代表熵为0
	if total == 0 {
		return 0
	}

	counts := make(map[int]float64)
	// 遍历标签，统计每个标签的数量
	for _, label := range y {
		counts[label]++
	}
	// 计算熵
	var ent float64
	// 遍历每个标签，计算熵
	for _, count := range counts {
		// 除以总体数量
		p := count / total
		// 然后求对数，最后乘以概率
		ent -= p * math.Log2(p)
	}
	// 返回熵
	return ent
}

// majorityVote 计算标签中的众数
//
//	例如 majorityVote([1, 2, 3, 1, 2, 3]) 返回 1
//	具体是通过统计每个标签的数量，然后返回数量最多的标签
func majorityVote(y []int) int {
	counts := make(map[int]int)
	// 遍历标签，统计每个标签的数量
	for _, label := range y {
		// 统计每个标签的数量
		counts[label]++
	}

	maxCount := -1
	majorityLabel := -1
	// 遍历每个标签的数量，找到数量最多的标签，因为数量最多的标签就是众数
	for label, count := range counts {
		if count > maxCount {
			maxCount = count
			majorityLabel = label
		}
	}
	return majorityLabel
}

// 用于操作切片和计算统计数据的辅助函数
//
//	例如 getColumn([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1) 返回 [2, 5, 8]
//	具体是通过遍历二维切片的每一行，然后获取某一行的某一列的值
//	然后将这些值放到一个新的切片中，最后返回这个新的切片
func getColumn(X [][]float64, index int) []float64 {
	// 获取二维切片的某一列
	column := make([]float64, len(X))
	// 遍历二维切片的每一行
	for i := range X {
		// 获取某一行的某一列的值
		column[i] = X[i][index]
	}
	return column
}

// unique 用于获取切片中的唯一值
//
//	例如 unique([1, 2, 3, 1, 2, 3]) 返回 [1, 2, 3]
//	具体是通过遍历切片，然后将切片中的值放到一个 map 中，最后将 map 中的值放到一个新的切片中
//	最后返回这个新的切片，这个新的切片中的值就是切片中的唯一值
func unique(values []float64) []float64 {
	seen := make(map[float64]bool)
	var result []float64
	// 遍历切片，将切片中的值放到一个 map 中
	for _, v := range values {
		// 如果切片中的值没有在 map 中，则将切片中的值放到 map 中
		if !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}
	return result
}

// splitDataset 用于划分数据集
//
//	例如 splitDataset([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1, 5) 返回 ([0], [1, 2])
//	具体是通过遍历数据集，然后根据特征的值来划分数据集
//	最后返回左子集和右子集的索引
func splitDataset(X [][]float64, featureIndex int, threshold float64) (left, right []int) {
	// 遍历数据集，然后根据特征的值来划分数据集
	for i, sample := range X {
		// 如果样本的特征值小于节点的阈值，则放到左子集
		if sample[featureIndex] < threshold {
			// 将索引放到左子集
			left = append(left, i)
		} else {
			// 如果样本的特征值大于等于节点的阈值，则放到右子集
			right = append(right, i)
		}
	}
	return
}

// subset 用于获取切片的子集
//
//	例如 subset([1, 2, 3, 4, 5], [0, 2, 4]) 返回 [1, 3, 5]
//	具体是通过遍历索引，然后根据索引获取切片中的值
//	最后将这些值放到一个新的切片中，最后返回这个新的切片
func subset[T any](data []T, indices []int) []T {
	var result []T
	// 遍历索引，然后根据索引获取切片中的值
	for _, index := range indices {
		// 将切片中的值放到一个新的切片中
		result = append(result, data[index])
	}
	return result
}

// Predict 对 X 中的每个样本进行预测
//
// 例如
//
//	dt := NewDecisionTree(3, 2)
//	X := [][]float64{
//		{0.1, 0.2, 0.3},
//		{0.4, 0.5, 0.6},
//	}
//	y := []int{0, 1}
//	dt.Fit(X, y)
//	yPred := dt.Predict(X)
//
// 返回预测的标签
func (dt *DecisionTree) Predict(X [][]float64) []int {
	yPred := make([]int, len(X))
	// 遍历 X 中的每个样本，然后对每个样本进行预测
	for i, x := range X {
		// 对单个样本进行预测
		yPred[i] = dt.predict(x, dt.root)
	}
	// 返回预测的标签
	return yPred
}

// predict 对单个样本进行预测
//
// 具体是通过递归的方式来预测样本的标签
// 如果节点的值不为-1，则返回节点的值
// 如果样本的特征值小于节点的阈值，则递归预测左子节点
// 如果样本的特征值大于等于节点的阈值，则递归预测右子节点
// 最后返回预测的标签
func (dt *DecisionTree) predict(x []float64, node *Node) int {
	// 如果节点的值不为-1，则返回节点的值
	if node.value != -1 {
		return node.value
	}
	// 如果样本的特征值小于节点的阈值，则递归预测左子节点
	if x[node.featureIndex] < node.threshold {
		// 递归预测左子节点
		return dt.predict(x, node.left)
	} else {
		return dt.predict(x, node.right)
	}
}
