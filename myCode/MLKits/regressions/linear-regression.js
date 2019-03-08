const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
	constructor(features, labels, options) {
		this.features = tf.tensor(features);
		this.labels = tf.tensor(labels);

		this.features = tf
			.ones([this.features.shape[0],1])
			.concat(this.features,1);

		this.options = Object.assign({ 
			learningRate: 0.1, 
			iterations: 1000 
		}, options);

		this.weights = tf.zeros([2,1]);

	}

	// gradientDescent() {
	// 	const currentGuessesForMPG = this.features.map(row => {
	// 		return this.a * row[0] + this.b;
	// 	});

	// 	const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
	// 		return guess - this.labels[i][0];
	// 	}))*2/this.labels.length;

	// 	const aSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
	// 		return -1*this.features[i][0]*(this.labels[i][0] - guess);
	// 	}))*2/this.labels.length;

	// 	this.a -= (aSlope * this.options.learningRate);
	// 	this.b -= (bSlope * this.options.learningRate);
	// }


	// New matrix based unified method (Features * ((Features * Weights) - Labels)) /n
	gradientDescent() {

		const currentGuesses = this.features.matMul(this.weights);
		const differences = currentGuesses.sub(this.labels);

		const slopes = this.features
			.transpose()
			.matMul(differences)
			.div(this.features.shape[0]);

		this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
	}

	train() {
		for (let i = 0; i < this.options.iterations; i++) {
			this.gradientDescent();
		}
	}

	test(testFeatures, testLabels) {
		testFeatures = tf.tensor(testFeatures); 
		testLabels = tf.tensor(testLabels);


		testFeatures = tf
			.ones([testFeatures.shape[0],1])
			.concat(testFeatures,1);

		const predictions = testFeatures.matMul(this.weights);

		const res = testLabels
			.sub(predictions)
			.pow(2)
			.sum()
			.get();

		const tot = testLabels
			.sub(testLabels.mean())
			.pow(2)
			.sum()
			.get();

		const r2 =  1 - (res/tot);

		console.log(`R2 is: ${r2}`);
	}
}

module.exports = LinearRegression;