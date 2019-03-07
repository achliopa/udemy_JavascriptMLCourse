const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
	constructor(features, labels, options) {
		this.features = features;
		this.labels = labels;
		this.options = Object.assign({ 
			learningRate: 0.1, 
			iterations: 1000 
		}, options);

		this.a = 0;
		this.b = 0;

	}

	train() {
		for (let i = 0; i < this.options.iterations; i++) {
			this.gradientDescent()
		}
	}

	gradientDescent() {
		const currentGuessesForMPG = this.features.map(row => {
			return this.a * row[0] + this.b;
		});

		const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
			return guess - this.labels[i][0];
		}))*2/this.labels.length;

		const aSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
			return -1*this.features[i][0]*(this.labels[i][0] - guess);
		}))*2/this.labels.length;

		this.a -= (aSlope * this.options.learningRate);
		this.b -= (bSlope * this.options.learningRate);
	}
}

module.exports = LinearRegression;