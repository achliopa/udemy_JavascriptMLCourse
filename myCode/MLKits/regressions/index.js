require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const loadCSV = require('./load-csv');

const LinearRegression = require('./linear-regression');

let {features, labels, testFeatures, testLabels } =   loadCSV('./cars.csv',{
	shuffle: true,
	splitTest: 50,
	dataColumns: ['horsepower','weight','displacement'],
	labelColumns: ['mpg']
});


const regression = new LinearRegression(features, labels, { 
	learningRate: .1,
	iterations: 3,
	batchSize: 10
});

regression.train();
regression.test(testFeatures, testLabels);

plot({
	x: regression.mseHistory.reverse(),
	xLabel: 'Iteration #',
	yLabel: 'Mean Square Error'
});

regression.predict([
	[120,2,380]
]).print();