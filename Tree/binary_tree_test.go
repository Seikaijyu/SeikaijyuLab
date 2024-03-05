package tree

import (
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBinaryTree(t *testing.T) {
	/// 因为二叉树中插入的值会自动排序，所以这里只需要测试是否排序正确即可
	/// 例如插入顺序为 10, 5, 15, 8, 3, 7, 20, 12, 18
	/// 那么中序遍历的结果就是 3, 5, 7, 8, 10, 12, 15, 18, 20
	list := []int{10, 5, 15, 8, 3, 7, 20, 12, 18}
	// 创建二叉树
	bt := NewBinaryTree[int]()
	// 插入节点
	for _, v := range list {
		bt.Insert(v)
	}
	// 排序
	sort.Ints(list)
	// 中序遍历
	bt.Each(func(idx int, value int) {
		assert.Equal(t, value, list[idx])
	})

	bt.Delete(3)
	list = []int{10, 5, 15, 8, 7, 20, 12, 18}
	sort.Ints(list)
	bt.Each(func(idx int, value int) {
		assert.Equal(t, value, list[idx])
	})
	assert.Equal(t, bt.Search(5), true)

}
