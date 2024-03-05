package ml

// 遗传算法实现
import (
	"math"
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
	Target           float64 // 目标
	PopulationSize   int     // 种群大小
	ChromosomeLength int     // 染色体长度
	MutationRate     float64 // 变异率
	CrossoverRate    float64 // 交叉率
	Generations      int     // 迭代次数
}

// 创建遗传算法实例
func NewGeneticAlgorithm(profile *GeneticAlgorithmProfile) *GeneticAlgorithm {
	// 创建一个新的随机数生成器实例，用于所有随机数生成
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)

	return &GeneticAlgorithm{
		profile:    profile,
		randSource: r, // 设置局部随机数生成器
	}
}

func (ga *GeneticAlgorithm) InitializePopulation() {
	ga.population = make([][]float64, ga.profile.PopulationSize)
	for i := range ga.population {
		ga.population[i] = make([]float64, ga.profile.ChromosomeLength)
		for j := range ga.population[i] {
			ga.population[i][j] = ga.randSource.Float64() * 100
		}
	}
}

func (ga *GeneticAlgorithm) CalculateFitness(chromosome []float64) float64 {
	sum := 0.0
	for _, gene := range chromosome {
		sum += gene
	}
	avg := sum / float64(len(chromosome))
	return 1 / (1 + math.Abs(avg-ga.profile.Target))
}

func (ga *GeneticAlgorithm) SelectParents() [][]float64 {
	fitnesses := make([]float64, ga.profile.PopulationSize)
	var totalFitness float64
	for i, chromosome := range ga.population {
		fitness := ga.CalculateFitness(chromosome)
		fitnesses[i] = fitness
		totalFitness += fitness
	}
	parents := make([][]float64, ga.profile.PopulationSize)
	for i := range parents {
		r := ga.randSource.Float64() * totalFitness
		sum := 0.0
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

func (ga *GeneticAlgorithm) Crossover(parent1, parent2 []float64) ([]float64, []float64) {
	if ga.randSource.Float64() < ga.profile.CrossoverRate {
		crossoverPoint := ga.randSource.Intn(ga.profile.ChromosomeLength - 1)
		child1 := append([]float64{}, parent1[:crossoverPoint]...)
		child1 = append(child1, parent2[crossoverPoint:]...)
		child2 := append([]float64{}, parent2[:crossoverPoint]...)
		child2 = append(child2, parent1[crossoverPoint:]...)
		return child1, child2
	}
	return parent1, parent2
}

func (ga *GeneticAlgorithm) Mutation(chromosome []float64) []float64 {
	for i := range chromosome {
		if ga.randSource.Float64() < ga.profile.MutationRate {
			chromosome[i] = ga.randSource.Float64() * 100
		}
	}
	return chromosome
}

func (ga *GeneticAlgorithm) Run() float64 {
	ga.InitializePopulation()
	for g := 0; g < ga.profile.Generations; g++ {
		newPopulation := make([][]float64, 0, ga.profile.PopulationSize)
		parents := ga.SelectParents()
		for i := 0; i < ga.profile.PopulationSize; i += 2 {
			parent1, parent2 := parents[i], parents[i+1]
			child1, child2 := ga.Crossover(parent1, parent2)
			child1 = ga.Mutation(child1)
			child2 = ga.Mutation(child2)
			newPopulation = append(newPopulation, child1, child2)
		}
		ga.population = newPopulation
	}
	bestFitness := 0.0
	bestIndex := 0
	for i, chromosome := range ga.population {
		fitness := ga.CalculateFitness(chromosome)
		if fitness > bestFitness {
			bestFitness = fitness
			bestIndex = i
		}
	}
	bestSolution := ga.population[bestIndex]
	sum := 0.0
	for _, gene := range bestSolution {
		sum += gene
	}
	return sum / float64(len(bestSolution))
}
