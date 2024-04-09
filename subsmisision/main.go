package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/sajari/regression"
)

// generateRandomData menghasilkan data acak untuk harga rumah
func generateRandomData(n int) ([][]float64, []float64) {
	var features [][]float64
	var labels []float64

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < n; i++ {
		// fitur-fitur acak
		area := rand.Float64() * 5000 // luas rumah
		bedrooms := rand.Float64() * 10 // jumlah kamar tidur
		bathrooms := rand.Float64() * 5 // jumlah kamar mandi
		distance := rand.Float64() * 50 // jarak ke pusat kota

		// harga rumah (label) yang dihasilkan menggunakan fungsi linear sederhana
		price := 1000*area + 5000*bedrooms + 3000*bathrooms - 200*distance + rand.Float64()*10000

		features = append(features, []float64{area, bedrooms, bathrooms, distance})
		labels = append(labels, price)
	}

	return features, labels
}

func main() {
	// Generate random data
	features, labels := generateRandomData(1000)

	// Membuat model regresi
	model := regression.NewLinearLeastSquares()

	// Menambahkan data ke model
	for i, feature := range features {
		if err := model.Train(feature, labels[i]); err != nil {
			log.Fatal(err)
		}
	}

	// Prediksi harga rumah baru
	newHouse := []float64{3000, 4, 3, 10} // Fitur rumah baru: luas 3000 sqft, 4 kamar tidur, 3 kamar mandi, jarak 10 mil
	predictedPrice, err := model.Predict(newHouse)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Harga rumah diprediksi: $%.2f\n", predictedPrice)

	// Membuat model klasifikasi sederhana
	classifier := regression.NewLogistic(regression.Binary)
	for i, feature := range features {
		// Jika harga rumah lebih dari $500.000, maka dianggap mahal
		isExpensive := labels[i] > 500000
		if err := classifier.Train(feature, isExpensive); err != nil {
			log.Fatal(err)
		}
	}

	// Prediksi apakah rumah baru adalah mahal atau tidak
	isExpensive, err := classifier.Predict(newHouse)
	if err != nil {
		log.Fatal(err)
	}

	if isExpensive > 0.5 {
		fmt.Println("Rumah tersebut diperkirakan mahal.")
	} else {
		fmt.Println("Rumah tersebut diperkirakan tidak mahal.")
	}
}
