package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/Seikaijyu/SeikaijyuLab/decision_tree_id3"
)

var randSource = rand.New(rand.NewSource(time.Now().UnixNano()))

// 生成随机数据，格式为
// [
//
//	[0.1, 0.2, 0.3],
//	[0.4, 0.5, 0.6],
//	...
//
// ]
func generateRandomData(rows, cols int, maxValue float64) [][]float64 {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
		for j := range data[i] {
			data[i][j] = randSource.Float64() * maxValue
		}
	}
	return data
}

// 将数据转换为二进制数据，小于等于阈值的为1，大于阈值的为0
func convertToBinaryData(data [][]float64, threshold float64) []int {
	binaryData := make([]int, len(data))
	for i, row := range data {
		if row[0] <= threshold {
			binaryData[i] = 1
		} else {
			binaryData[i] = 0
		}
	}
	return binaryData
}

// 计算准确率
func calculateAccuracy(predictions, labels []int) float64 {
	if len(predictions) != len(labels) {
		return 0.0
	}
	correct := 0
	for i := range predictions {
		if predictions[i] == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(predictions))
}

func main() {
	// 生产训练数据，格式为
	// [
	//	[0.1],
	//	[0.2],
	//	...
	// ]
	X_train := generateRandomData(1000, 1, 100)
	// 将训练数据转换为二进制数据
	Y_train := convertToBinaryData(X_train, 50)

	// 创建决策树
	tree := decision_tree_id3.NewDecisionTree(3, 2)

	// 训练决策树
	tree.Fit(X_train, Y_train)
	// 生成测试数据
	X_test := generateRandomData(100, 1, 100)
	// 将测试数据转换为二进制数据
	Y_test := convertToBinaryData(X_test, 50)
	// 预测
	predictions := make([]int, len(X_test))
	for i, x := range X_test {
		predictions[i] = tree.Predict([][]float64{x})[0]
	}

	// 计算准确率
	accuracy := calculateAccuracy(predictions, Y_test)
	fmt.Printf("Accuracy: %f\n", accuracy)
}
