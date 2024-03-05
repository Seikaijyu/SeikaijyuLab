package ml

// 遗传算法实现
import (
	"math/rand"
	"time"
)

type GeneticAlgorithm struct {
	profile    *GeneticAlgorithmProfile
	population [][]float64
	randSource *rand.Rand // 添加一个局部随机数生成器
}

// 遗传算法配置
type GeneticAlgorithmProfile struct {
	PopulationSize    int                                 // 种群大小，代表一对个体
	ChromosomeLength  int                                 // 染色体长度，代表一对染色体
	MutationRate      float64                             // 变异率
	CrossoverRate     float64                             // 交叉率
	Generations       int                                 // 迭代次数
	GeneRange         float64                             // 基因范围
	ChromosomeValFunc func(chromosome []float64) float64  // 染色体求值函数
	FitnessFunc       func(chromosomeVal float64) float64 // 适应度函数
}

// 创建一个新的遗传算法实例
//
//	遗传算法是一种模拟自然界生物进化过程的优化算法，它通过模拟生物的自然选择、交叉和变异等操作，寻找最优解。
//	遗传算法的基本思想是：通过随机生成一些个体的初始种群，然后根据每个个体的适应度函数来评价个体的适应度，适应度高的个体有更大的机会被选中。
//	然后，通过交叉和变异等操作来产生新的个体，新个体的适应度同样由适应度函数来评价。通过这样的迭代，最终产生一个适应度较高的个体，这个个体就是所求的最优解。
//	遗传算法的优点是：适应性强，对于解决复杂的、非线性的优化问题有很好的效果。
//	遗传算法的缺点是：算法的参数设置较多，运行时间较长，收敛速度较慢。
//	遗传算法的应用：遗传算法主要用于解决复杂的、非线性的优化问题，如：工程优化、生产调度、图形识别、机器学习等问题。
//	一般来说，遗传算法的参数设置需要根据具体问题的特点来设置，这些参数包括：种群大小、染色体长度、变异率、交叉率、迭代次数等。
//	例如，对于某个具体的问题，可以通过遗传算法来求解该问题的最优解，具体步骤如下：
//	1. 定义目标函数：首先定义一个目标函数，该函数是问题的数学模型，通过该函数来评价个体的适应度。
//	2. 初始化种群：随机生成一些个体作为初始种群，种群的大小、染色体的长度等需要根据具体问题来设置。
//	3. 评价个体：通过目标函数来评价种群中每个个体的适应度，适应度函数的设计需要根据具体问题来设置。
//	4. 选择个体：通过适应度函数来选择适应度较高的个体，适应度较高的个体有更大的机会被选中。
//	5. 交叉和变异：通过交叉和变异等操作来产生新的个体，新个体的适应度同样由适应度函数来评价。
//	6. 迭代更新：通过迭代更新的方式来产生新的个体，最终产生一个适应度较高的个体，这个个体就是所求的最优解。
func NewGeneticAlgorithm(profile *GeneticAlgorithmProfile) *GeneticAlgorithm {
	// 创建一个新的随机数生成器实例，用于所有随机数生成
	src := rand.NewSource(time.Now().UnixNano())
	// 通过随机数生成器实例创建一个新的随机数生成器
	r := rand.New(src)
	// 为了方便选择父代，将种群大小和染色体长度设置为偶数
	profile.ChromosomeLength = profile.ChromosomeLength * 2
	profile.PopulationSize = profile.PopulationSize * 2
	return &GeneticAlgorithm{
		profile:    profile,
		randSource: r, // 设置局部随机数生成器
	}
}

// 初始化种群
func (ga *GeneticAlgorithm) InitializePopulation() {
	// 创建种群
	ga.population = make([][]float64, ga.profile.PopulationSize)
	// 初始化种群中的每个染色体
	for i := range ga.population {
		// 为每个生物体创建一个染色体
		ga.population[i] = make([]float64, ga.profile.ChromosomeLength)
		// 通过随机数初始化染色体中的每个基因
		for j := range ga.population[i] {
			// 随机生成一个基因值
			ga.population[i][j] = ga.randSource.Float64() * ga.profile.GeneRange
		}
	}
}

// 计算适应度
func (ga *GeneticAlgorithm) CalculateFitness(chromosome []float64) float64 {
	// 先求出染色体的值，然后再通过适应度函数来评价个体的适应度
	return ga.profile.FitnessFunc(ga.profile.ChromosomeValFunc(chromosome))
}

// 选择父代
//
//	首先计算种群中每个个体的适应度，然后根据适应度函数来评价个体的适应度，适应度高的个体有更大的机会被选中。
//	然后，通过轮盘赌法来选择父代，轮盘赌法的基本思想是：随机生成一个数，该数的范围是种群中所有个体的适应度之和，然后根据适应度之和来选择父代。
//	通过轮盘赌法，适应度较高的个体有更大的机会被选中，适应度较低的个体有更小的机会被选中。
//	轮盘赌法的一个重要参数是选择概率，选择概率的大小直接影响了选择父代的频率，选择概率越大，选择父代的频率越高。
func (ga *GeneticAlgorithm) SelectParents() [][]float64 {
	fitnesses := make([]float64, ga.profile.PopulationSize)
	var totalFitness float64
	for i, chromosome := range ga.population {
		fitness := ga.CalculateFitness(chromosome)
		fitnesses[i] = fitness
		totalFitness += fitness
	}
	// 通过轮盘赌法选择父代
	parents := make([][]float64, ga.profile.PopulationSize)
	// 遍历种群中的每个个体
	for i := range parents {
		// 随机生成一个数，该数的范围是种群中所有个体的适应度之和
		r := ga.randSource.Float64() * totalFitness
		sum := 0.0
		// 遍历种群中的每个个体的适应度，计算适应度之和，直到适应度之和大于等于随机数r，该个体就是被选中的父代
		for j, fitness := range fitnesses {
			sum += fitness
			if sum >= r {
				parents[i] = ga.population[j]
				break
			}
		}
	}
	return parents
}

// 交叉选择
//
//	首先随机生成一个交叉点，然后将父代的染色体分割成两部分，然后将两个父代的染色体的一部分交换，产生两个子代。
//	假设父代的染色体分别是A、B，交叉点是3，那么可以产生两个子代：A1 + B2、B1 + A2。
//	交叉选择的一个重要参数是交叉率，交叉率的大小直接影响了交叉选择的频率，交叉率越大，交叉选择的频率越高。
func (ga *GeneticAlgorithm) Crossover(parent1, parent2 []float64) ([]float64, []float64) {
	// 如果随机数小于交叉率，则进行交如果随机数小于交叉率，则进行交叉叉操作
	if ga.randSource.Float64() < ga.profile.CrossoverRate {
		// 随机生成一个交叉点，交叉点的范围是0到染色体的长度之间的一个随机数
		crossoverPoint := ga.randSource.Intn(ga.profile.ChromosomeLength - 1)
		// 将父代的染色体分割成两部分，然后将两个父代的染色体的一部分交换，产生两个子代
		child1 := append([]float64{}, parent1[:crossoverPoint]...)
		child1 = append(child1, parent2[crossoverPoint:]...)
		child2 := append([]float64{}, parent2[:crossoverPoint]...)
		child2 = append(child2, parent1[crossoverPoint:]...)
		// 返回两个子代
		return child1, child2
	}
	// 如果随机数大于交叉率，则不进行交叉操作，直接返回父代
	return parent1, parent2
}

// 变异操作
//
//	随机生成一个数，如果该数小于变异率，则对染色体中的某个基因进行变异。
//	变异操作是遗传算法中的一个重要操作，它通过变异操作来产生新的个体，新个体的适应度同样由适应度函数来评价。
//	变异操作的一个重要参数是变异率，变异率的大小直接影响了变异操作的频率，变异率越大，变异操作的频率越高。
func (ga *GeneticAlgorithm) Mutation(chromosome []float64) []float64 {
	// 遍历染色体中的每个基因
	for i := range chromosome {
		// 随机生成一个数，如果该数小于变异率，则对染色体中的某个基因进行变异
		if ga.randSource.Float64() < ga.profile.MutationRate {
			// 随机生成一个基因值
			chromosome[i] = ga.randSource.Float64() * ga.profile.GeneRange
		}
	}
	return chromosome
}

// 运行遗传算法
func (ga *GeneticAlgorithm) Start() float64 {
	// 初始化种群
	ga.InitializePopulation()
	// 通过迭代更新的方式来产生新的个体，最终产生一个适应度较高的个体，这个个体就是所求的最优解。
	for g := 0; g < ga.profile.Generations; g++ {
		// 初始化一个新的种群
		newPopulation := make([][]float64, 0, ga.profile.PopulationSize)
		// 选择父代
		parents := ga.SelectParents()
		// 进行交叉和编译操作
		// 每次循环都会处理两个父代个体（parent1 和 parent2）
		// 并生成两个子代个体（child1 和 child2）。
		// 因此，每次循环后，索引 i 需要增加 2，以便在下一次循环中处理下一对父代个体
		for i := 0; i < ga.profile.PopulationSize; i += 2 {
			parent1, parent2 := parents[i], parents[i+1]
			child1, child2 := ga.Crossover(parent1, parent2)
			child1, child2 = ga.Mutation(child1), ga.Mutation(child2)
			// 将两个子代添加到新的种群中
			newPopulation = append(newPopulation, child1, child2)
		}
		// 更新种群
		ga.population = newPopulation
	}
	// 最佳适应度
	bestFitness := 0.0
	// 最佳个体索引
	bestIndex := 0
	// 遍历种群中的每个个体
	for i, chromosome := range ga.population {
		// 计算适应度
		fitness := ga.CalculateFitness(chromosome)
		// 如果适应度大于最佳适应度，则更新最佳适应度和最佳个体索引
		if fitness > bestFitness {
			bestFitness = fitness
			bestIndex = i
		}
	}
	// 返回最佳适应度生物的染色体求值
	return ga.profile.ChromosomeValFunc(ga.population[bestIndex])
}
