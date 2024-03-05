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

// 递归插入
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

// 中序遍历
func (bt *BinaryTree[T]) InOrderTraversal() []T {
	return bt.inOrderTraversalRecursive(bt.Root)
}

// 递归中序遍历
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
func NewBinaryTree[T constraints.Integer | constraints.Float]() *BinaryTree[T] {
	return &BinaryTree[T]{}
}
