package tree

// 非平衡二叉树实现

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

// 节点
type Node[T constraints.Integer | constraints.Float] struct {
	// 当前节点的值
	Value T
	// 左节点
	Left *Node[T]
	// 右节点
	Right *Node[T]
}

// 二叉树
type BinaryTree[T constraints.Integer | constraints.Float] struct {
	// 根节点
	Root *Node[T]
}

// 插入节点
func (bt *BinaryTree[T]) Insert(value T) {
	if bt.Root == nil {
		// 如果根节点为空，直接插入
		bt.Root = &Node[T]{Value: value}
	} else {
		// 否则递归插入
		bt.insertRecursive(bt.Root, value)
	}
}

// 查找节点
func (bt *BinaryTree[T]) Search(value T) bool {
	return bt.searchRecursive(bt.Root, value)
}

// 递归查找
//
//	查找节点，如果当前节点为空，返回false
//	如果当前节点的值等于要查找的值，返回true
//	如果要查找的值小于当前节点的值，递归查找左节点
//	如果要查找的值大于当前节点的值，递归查找右节点
func (bt *BinaryTree[T]) searchRecursive(node *Node[T], value T) bool {
	if node == nil {
		return false
	}
	if value < node.Value {
		return bt.searchRecursive(node.Left, value)
	} else if value > node.Value {
		return bt.searchRecursive(node.Right, value)
	} else {
		return true
	}
}

// 删除节点
func (bt *BinaryTree[T]) Delete(value T) {
	bt.deleteRecursive(bt.Root, value)
}

// 递归删除
//
//	删除节点，如果当前节点为空，返回nil
//	如果要删除的节点小于当前节点的值，递归删除左节点
//	如果要删除的节点大于当前节点的值，递归删除右节点
//	如果当前节点的值等于要删除的值，如果左节点为空，返回右节点
//	如果右节点为空，返回左节点
//	如果左右节点都不为空，找到右节点的最小值，替换当前节点的值，递归删除右节点的最小值
func (bt *BinaryTree[T]) deleteRecursive(node *Node[T], value T) *Node[T] {
	if node == nil {
		return nil
	}
	if value < node.Value {
		node.Left = bt.deleteRecursive(node.Left, value)
	} else if value > node.Value {
		node.Right = bt.deleteRecursive(node.Right, value)
	} else {
		if node.Left == nil {
			return node.Right
		} else if node.Right == nil {
			return node.Left
		}
		node.Value = bt.minValue(node.Right)
		node.Right = bt.deleteRecursive(node.Right, node.Value)
	}
	return node
}

// 查找最小值
//
//	查找最小值，如果左节点为空，返回当前节点的值
//	否则递归查找左节点
func (bt *BinaryTree[T]) minValue(node *Node[T]) T {
	minValue := node.Value
	for node.Left != nil {
		minValue = node.Left.Value
		node = node.Left
	}
	return minValue
}

// 递归插入
//
//	插入节点，如果小于当前节点的值，插入到左节点
//	如果大于当前节点的值，插入到右节点
//	如果等于当前节点的值，提示已经存在
func (bt *BinaryTree[T]) insertRecursive(node *Node[T], value T) {
	if value < node.Value {
		// 如果小于当前节点的值，插入到左节点
		if node.Left == nil {
			node.Left = &Node[T]{Value: value}
		} else {
			bt.insertRecursive(node.Left, value)
		}
	} else if value > node.Value {
		// 如果大于当前节点的值，插入到右节点
		if node.Right == nil {
			node.Right = &Node[T]{Value: value}
		} else {
			bt.insertRecursive(node.Right, value)
		}
	} else {
		// 如果等于当前节点的值，提示已经存在
		fmt.Println("Value is already present in the tree")
	}
}

// 遍历节点
func (bt *BinaryTree[T]) Each(f func(idx int, value T)) {
	for i, v := range bt.InOrderTraversal() {
		f(i, v)
	}
}

// 中序遍历
func (bt *BinaryTree[T]) InOrderTraversal() []T {
	return bt.inOrderTraversalRecursive(bt.Root)
}

// 递归中序遍历
//
//	中序遍历的顺序是：左节点 -> 当前节点 -> 右节点
//	递归遍历，如果当前节点为空，返回空数组
//	否则递归遍历左节点，当前节点的值，递归遍历右节点
func (bt *BinaryTree[T]) inOrderTraversalRecursive(node *Node[T]) []T {
	if node != nil {
		l := bt.inOrderTraversalRecursive(node.Left)
		reslut := append(l, node.Value)
		r := bt.inOrderTraversalRecursive(node.Right)
		return append(reslut, r...)
	}
	return []T{}
}

// 新建一个二叉树
//
//	二叉树是一种树形结构，它的每个节点最多有两个子节点，分别称为左子节点和右子节点。
//	二叉树的每个节点包含三个信息：节点的值、左子节点和右子节点。
//	二叉树的基本操作包括：插入节点、删除节点、查找节点、遍历节点等。
func NewBinaryTree[T constraints.Integer | constraints.Float]() *BinaryTree[T] {
	return &BinaryTree[T]{}
}
