package ml

import (
	"math"
	"testing"
)

func TestGeneticAlgorithm(t *testing.T) {
	want := 25
	ga := NewGeneticAlgorithm(&GeneticAlgorithmProfile{
		PopulationSize:   50,
		ChromosomeLength: 5,
		MutationRate:     0.01,
		CrossoverRate:    0.7,
		Generations:      1000,
		GeneRange:        100,
		// 计算适应度
		FitnessFunc: func(chromosomeVal float64) float64 {
			return 1 / (1 + math.Abs(chromosomeVal-float64(want)))
		},
		// 计算染色体的值
		ChromosomeValFunc: func(chromosome []float64) float64 {
			var sum float64
			for _, v := range chromosome {
				sum += v
			}
			return sum / float64(len(chromosome))
		},
	})
	got := int(math.Round(ga.Run()))
	if got != want {
		t.Errorf("Run() = %v, want %v", got, want)
	} else {
		t.Log("Run() = ", got)
	}

}
