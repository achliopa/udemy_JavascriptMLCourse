require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const loadCSV = require('../data/load-csv');

const LogisticRegression = require('./logistic-regression');

let {features, labels, testFeatures, testLabels } =   loadCSV('../data/cars.csv',{
	shuffle: true,
	splitTest: 50,
	converters: {
		passedemissions: (value) => {
			return value === 'TRUE' ? 1 : 0;
		}
	},
	dataColumns: ['horsepower','displacement','weight'],
	labelColumns: ['passedemissions']
});

const regression = new LogisticRegression(features,labels, {
	learningRate: 0.5,
	iterations: 100,
	batchSize: 10,
	decisionBoundary: .52
});

regression.train();
console.log(`Accuracy ${regression.test(testFeatures, testLabels )*100}%`);

plot({
	x: regression.costHistory.reverse()
});