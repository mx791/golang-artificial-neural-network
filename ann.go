package main

import (
	"fmt"
	"math/rand"
	"math"
	"time"
	"sync"
)

func getDataX() [][]float64 {
	data := make([][]float64, 200)
	for i := 0; i<100; i++ {
		data[i] = make([]float64, 10)
		start := rand.Float64()
		for e:=0; e<10; e++ {
			data[i][e] = start + rand.Float64()*0.2 + float64(e)*0.02
		}
	}
	for i := 100; i<200; i++ {
		data[i] = make([]float64, 10)
		start := rand.Float64()
		for e:=0; e<10; e++ {
			data[i][e] = start + rand.Float64()*0.2 - float64(e)*0.02
		}
	}
	return data
}

func getDataY() [][]float64 {
	data := make([][]float64, 200)
	for i := 0; i<100; i++ {
		data[i] = make([]float64, 2)
		data[i][0] = 1.0
		data[i][1] = 0.0
	}
	for i := 100; i<200; i++ {
		data[i] = make([]float64, 2)
		data[i][1] = 1.0
		data[i][0] = 0.0
	}
	return data
}

func ternary(condition bool, value1 float64, value2 float64) float64 {
	if condition {
		return value1
	}
	return value2
}

type Layer struct {
	units int
	activation string
	useBias bool
}

// génère une liste de matrice de poids par rapport aux spec des layers
func builLayers(inputSize int, layers []Layer, useRandom bool, baseValue float64) [][][]float64 {
	weights := make([][][]float64, len(layers))
	lastSize := inputSize
	for layer:=0; layer<len(layers); layer++ {
		weights[layer] = make([][]float64, layers[layer].units)
		for out:=0; out<layers[layer].units; out++ {
			weights[layer][out] = make([]float64, int(ternary(layers[layer].useBias, float64(lastSize + 1), float64(lastSize))))
			for value:=0; value<len(weights[layer][out]); value++ {
				weights[layer][out][value] = ternary(useRandom, 0.5 - rand.Float64(), baseValue)
			}
		}
		lastSize = layers[layer].units
	}
	return weights
}

// fonction d'activation
func activation(name string, values []float64) []float64 {
	ret := make([]float64, len(values))
	sum := 0.0
	if name == "softmax" {
		for i:=0; i<len(values); i++ {
			sum += math.Exp(values[i])
		}
	}
	for i:=0; i<len(values); i++ {
		if name == "sigmoid" {
			ret[i] = 1.0 / (1.0 + math.Exp(-values[i]))
		} else if name == "softmax" {
			ret[i] = math.Exp(values[i]) / sum
		} else if name == "relu" {
			ret[i] = ternary(values[i] > 0.0, values[i], values[i]*0.05)
		} else {
			ret[i] = values[i]
		}
	}
	return ret
}

// dérivé de la fonction d'activation
func dactivation(name string, values []float64) []float64 {
	ret := make([]float64, len(values))
	sum := 0.0
	if name == "softmax" {
		for i:=0; i<len(values); i++ {
			sum += math.Exp(values[i])
		}
	}
	for i:=0; i<len(values); i++ {
		if name == "sigmoid" {
			ret[i] = values[i] * (1.0 - values[i])
		} else if name == "relu" {
			ret[i] = ternary(values[i] > 0.0, 1.0, 0.0)
		} else if name == "softmax" {
			ret[i] = 1.0 / sum
		} else {
			ret[i] = values[i]
		}
	}
	return ret
}

// cliping de gradient pour l'empêcher de dépasser un certain seuil
func controlGradient(value float64) float64 {
	CLIPING_THRESOLD := 0.5
	if math.Abs(value) > CLIPING_THRESOLD {
		return value / math.Abs(value)
	}
	return value
}

// variables pour la parallelisation
var wg sync.WaitGroup
var loss = 0.0
var goodClassified = 0
var allClassified = 0

// forward + backward pour une ligne du dataset
func threadIter(x []float64, y []float64, weights [][][]float64, gradient [][][]float64, layers []Layer) {
	defer wg.Done()
	layersValues := make([][]float64, len(layers) + 1)
	lastInput := x
	layersValues[0] = x
	for layerId:=0; layerId<len(layers); layerId++ {
		layerOutput := make([]float64, layers[layerId].units)
		for layerOutputId:=0; layerOutputId<len(layerOutput); layerOutputId++ {
			unitValue := 0.0
			for lastInputId:=0; lastInputId<len(lastInput); lastInputId++ {
				unitValue += lastInput[lastInputId] * weights[layerId][layerOutputId][lastInputId]
			}
			if layers[layerId].useBias {
				unitValue += weights[layerId][layerOutputId][len(lastInput)]
			}
			layerOutput[layerOutputId] = unitValue
		}
		layerOutput = activation(layers[layerId].activation, layerOutput)
		layersValues[layerId+1] = layerOutput
		lastInput = layerOutput
	}

	// loss
	lossVector := make([]float64, len(lastInput))
	dact := dactivation(layers[len(layers)-1].activation, lastInput)
	for lossId:=0; lossId<len(lastInput); lossId++ {
		lossVector[lossId] = (y[lossId] - lastInput[lossId]) * dact[lossId]
		loss += math.Pow(y[lossId] - lastInput[lossId], 2.0)
		if math.Abs(y[lossId] - lastInput[lossId]) < 0.5 {
			goodClassified += 1
		}
		allClassified += 1
	}

	// backward
	for layerId:=len(layers)-1; layerId>=0; layerId-- {
		newLossVect := make([]float64, len(layersValues[layerId]))
		derivatePreviousLayer := dactivation(layers[layerId].activation, layersValues[layerId])
		for newLossId:=0; newLossId<len(newLossVect); newLossId++ {
			for oldLossId:=0; oldLossId<len(layersValues[layerId+1]); oldLossId++ {
				newLossVect[newLossId] += lossVector[oldLossId] * weights[layerId][oldLossId][newLossId] * derivatePreviousLayer[newLossId]
				gradient[layerId][oldLossId][newLossId] += lossVector[oldLossId] * layersValues[layerId][newLossId]						
			}
		}
		if layers[layerId].useBias {
			for outId:=0; outId<len(lossVector); outId++ {
				gradient[layerId][outId][len(derivatePreviousLayer)] += lossVector[outId]
			}
		}
		lossVector = newLossVect
	}
}

// constantes d'entrainement
var ITERS = 300
var LEARN_RATE = 0.1
var BATCH_SIZE = 10

var b1 = 0.9
var b2 = 0.999
var eps = 0.0000001

func trainAnn(inputSize int, layers []Layer, x [][]float64, y [][]float64) ANN {

	weights := builLayers(inputSize, layers, true, 0.0)
	gradient := builLayers(inputSize, layers, false, 0.0)
	mouvingAvg := builLayers(inputSize, layers, false, 0.0)
	squaredMouvingAvg := builLayers(inputSize, layers, false, 0.0)

	updates := 0

	batch_count := len(x) / BATCH_SIZE
	
	for it:=0; it<ITERS; it++ {
		loss = 0.0
		goodClassified = 0
		allClassified = 0

		debut := time.Now()
		for batchId:=0; batchId<batch_count; batchId++ {

			for itm:=batchId*BATCH_SIZE; itm<(batchId+1)*BATCH_SIZE; itm++ {
				wg.Add(1)
				go threadIter(x[itm], y[itm], weights, gradient, layers)
			}
			wg.Wait()
			for i:=0; i<len(weights); i++ {
				for e:=0; e<len(weights[i]); e++ {
					for a:=0; a<len(weights[i][e]); a++ {
						grad := gradient[i][e][a] / float64(BATCH_SIZE)
						mChap := mouvingAvg[i][e][a] / (1.0 - math.Pow(b1, float64(updates)))
						vChap := math.Sqrt(squaredMouvingAvg[i][e][a] / (1.0 - math.Pow(b2, float64(updates)))) + eps
						admaRate := ternary(updates >= 1, mChap / vChap * LEARN_RATE, 1.0)
						mouvingAvg[i][e][a] = mouvingAvg[i][e][a]*b1 + (1.0 - b1)*grad
						squaredMouvingAvg[i][e][a] = squaredMouvingAvg[i][e][a] * b2 + (1.0 - b2) * math.Pow(grad, 2.0)
						weights[i][e][a] += controlGradient(grad * math.Abs(admaRate))
						gradient[i][e][a] = 0.0
					}
				}
			}
			updates += 1
		}
		
		fin := time.Now()
		loss = loss / float64(len(x)) / float64(len(y[0]))
		fmt.Println(it, "loss", loss, "acc", float64(goodClassified)/float64(allClassified), fin.Sub(debut))
	}

	return ANN{inputSize: inputSize, weights: weights, layers: layers}
}

// affiche à l'utilisateur des informations sur le modèles
func (net ANN) describe() {
	fmt.Println("---------------------")
	fmt.Println("Description du modèle")
	fmt.Println("Taille de l'entrée:", net.inputSize)
	params := 0
	prev := net.inputSize
	for i:=0; i<len(net.layers); i++ {
		if net.layers[i].useBias {
			params += net.layers[i].units * (prev + 1)
		} else {
			params += net.layers[i].units * prev
		}
		prev = net.layers[i].units
		fmt.Println("Dense layer", net.layers[i].units, "units", net.layers[i].activation)
	}
	fmt.Println("---------------------")
	fmt.Println(params, "paramètres")
	fmt.Println("---------------------")
}

type ANN struct {
	inputSize int
	weights [][][]float64
	layers []Layer
}

// réarange aléatoirement un dataset
func shuffle(x [][]float64, y [][]float64) {
	for i:=0; i<len(x); i++ {
		newId := rand.Intn(len(x))
		xTmp := x[newId]
		yTmp := y[newId]
		x[newId] = x[i]
		y[newId] = y[i]
		x[i] = xTmp
		y[i] = yTmp
	}
}

func main() {
	x := getDataX()
	y := getDataY()
	shuffle(x, y)
	layer1 := Layer{units: 5, activation: "relu", useBias: true}
	//layer2 := Layer{units: 15, activation: "relu", useBias: true}
	layer3 := Layer{units: 2, activation: "softmax", useBias: true}
	net := trainAnn(10, []Layer{layer1, layer3}, x, y)
	net.describe()
}
