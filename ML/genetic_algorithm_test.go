package ml

import (
	"math"
	"testing"
)

func TestRun(t *testing.T) {
	want := 90
	ga := NewGeneticAlgorithm(&GeneticAlgorithmProfile{
		Target:           float64(want),
		PopulationSize:   100,
		ChromosomeLength: 10,
		MutationRate:     0.01,
		CrossoverRate:    0.7,
		Generations:      1000,
	})

	got := int(math.Round(ga.Run()))
	if got != want {
		t.Errorf("Run() = %v, want %v", got, want)
	}
}
