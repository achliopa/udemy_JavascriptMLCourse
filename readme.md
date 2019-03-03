# Udemy Course: Machine Learning with Javascript by Stephen Grider

* [Course Link](https://www.udemy.com/machine-learning-with-javascript/)
* [Course Repo](https://github.com/StephenGrider/MLCasts)

## Section 1 - What is Machine Learning

### Lecture 2 - Solving Machine Learning Problems

* ML can do predicions based on historical data. e.g if it rain 240mm what the flood damage will there be
* The ML Problem Solving Process
	* Identify data that is relevant to the problem
	* Assemble a set of data related to the problem we're trying to solve
	* Decide on the type of output we are predicting
	* Based on type of output, pick an algorithm that will determine a correlations between our 'features' and 'labels'
	* Use model generated by algorithm to make a prediction
* For our example problem 'annual rainfall' is the feature or independent variable. Flood damage costs is the dependent variable 'label'
* in this course we will see how we can collect data format it and prepare it for ML
* we will assume we have no data. data can come from videos, newspapers, websites
* flood repair spending can come from budget reports. 
* all is assembled in a table

### Lecture 3 - A Complete Walkthrough

* Value of Labels are discrete set? Classification
* Value of Labes is continuous? Regression
* the example problem about floods is a regression problem (we predict money)
* Some takeaway points
	* Features are categories of data points that affect the value of labels
	* Datasets almost always need cleanup and formatting
	* Regression for continuous vals, Classification for diescrete
	* Many many different ML algorithms exist each with pros and cons anf or different type of problems.
	* models relate the value of features to the value of labels

### Lecture 4 - App Setup

* we go to [Github Starter Projects](https://github.com/StephenGrider/MLKits) and download them in our course workspace
* we go to '/MLKits/plinko'
* we open 'index.html'

### Lecture 5 - Problem Outline

* what we see is a game that puts (randomly?) disks in buckets increasing their counter.
* we will use it to build a dataset trying to predict in which bucket the ball will end to given the drop point

### Lecture 6 - Identifying Relevant Data

* apart from the priamry feature (drop position) we identify 2 minor features. ball size and bonciness. both are randonly selected between a configurable range of values

### Lecture 7 - Dataset Structures

* the app offers a lot of ways to record our tries and collect data (dataset)
* In JS datasets can be:
	* Arrays of objects eg `[{dropPosition: 300, bonciness: 0.4, ballSize: 16, bucket: 4}]`
	* Nested Arrays [[300,0,4,16,4],[350,0.4,35,5],[416,0.4,16,4]]. Indices are Important
* we open 'score.js'. it has 2 method signatures one to collect the data set and one to run the analyzsis to predict

### Lecture 8 - Recording Observation Data

* we flesh out 'onScoreUpdate' a callback  called everytime a round ends to collect data
```
const outputs = [];

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition,bounciness,size,bucketLabel]);
  console.log(outputs);
}
```
* we test and see an evergroing nested array

### Lecture 9 - What Type of Problem?

* our problem is clearly a Classification
* our classification algorithm of choice is K-Nearest Neighbor (knn) is based on clustering

## Section 2 - Algorithm Overview

### Lecture 10 - How K-Nearest Neighbor Works

* Thought Experiment: what wqould happen if we dropped the ball 10 times from almost the same spot (300px)? 
* K-Nearest Neighbor (with one independent variable)
	* Drop a ball a bunch of times all around the board, record which bucket it goes to
	* For each observation subtract drop point from 300px, take absolute value
	* Sort ther results from least to greatest
	* Look at the 'k' top  records. What was the most common bucket?
	Whichever bucket came up most frequently is the one ours will probably go to

### Lecture 11 - Lodash Review

* we will use [lodash](https://lodash.com/docs/). a JS utility library with lots of methods for arrays, objects etc
* we use JS interpreter tool [JSPlaygrounds](https://stephengrider.github.io/JSPlaygrounds/) to experiment with code
* if we have a 4x2 nested array and we want to sort it by the row second element we can use stortBy selecting the element based on which we want to sort
```
const numbers = [[10,5],[17,2],[34,1],[60,-5]];
const sorted = _.sortBy(numbers, row => row[1])
```
* we map through the sorted array to extract second element `const mapped = _.map(sorted, row => row[1]);`
* we see a pattern forming where we use the previous result as input for next method .... function chaining maybe??? '_.chain' allows to chain lodash methods passing in result
```
_.chain(numbers)
	.sortBy(row=>row[1])
	.map(row=>row[1])
	.value();
```
* note that we omit the first argument as its implicitly passe don. vlaue() stops the chain adn returns the result

### Lecture 12 - Implementing KNN

* we start implementing KNN in JS using lodash in JSPlaygrounds
* we start by defining a hardcoded dataset as nested array
```
const outputs = [
	[10, .5, 16, 1].
  [200, .5, 16, 4],
  [350, .5, 16, 4],
  [600, .5, 16, 5]
];
```
* for e ach observation, we will subtract droppoint from 300px, and take the absolute value. we will use map. our result table will have to datapoints per observation. droppoint and bucket
* we put our distance calculation in a helper
```
const predictionPoint = 300;

const distance = (point) => {
	return Math.abs(point - predictionPoint);
};
```
* we implement step 1 
```

_.chain(outputs)
  .map(row => [distance(row[0]),row[3]])
```
* step 2 is to sort results from least to greatest. we use sortBy inthe chain `.sortBy(row => row[0])`
* step 3 is to look at top 'k' records. whats the most common bucket??? we use 'slice'. for k=3 `.slice(0,3)`

### Lecture 13 - Finishing KNN Implementation

* step 4 is to look in these k top records. whats the most common bucket? we will use lodash 'countBy' counting the records that meet some criteria. what we get is an object with key value pairs `.countBy(row => row[1])` => '{"1":1,"4":2}'
* next we will use lodash `.toPairs()` to convert the object to nested array
* we use sortBy to sort based on second column (occurencies). most will go to bottom
* i need t get the last element using lodash `.last()` and then `.first()` to get first element (bucket)
* then we use `.parseInt()` to turn string to bucket and `.value()` to terminate chain

### Lecture 14 - Testing the Algorithm

* we take the code and move it in the runAnalysis function in 'score.js'
* we cp distance function in the js file under analysis method, we cp globals on top and lodash chain in the runAnalysis method. we import lodash
* we test by running the game for multiple balls to fill our data set. then we 'analyze' to see in which bucket it will fall if we drop it from point 300. this will run our js knn method
* to test the result we 'reset' and drop 100 balls at 300
* we see our algorithm is way off

### Lecture 15 - Interpreting bad Results

* Steps after realising we have bad results:
	* Adjust the parameters of the analysis
	* Add more features to explain the analysis (bounciness, ball size)
	* Change the prediction point
	* Accept the fact there isn't a good correlation
* we will try out different k and rerun the analysis to see if it has effect on accuracy

### Lecture 16 - Test and Training Data

* to improve the algo we need to have a good way to compare accuracy with different settings. e.g for many different prediction points
* the way to find the ideal K
	* record a bunch of data points
	* split that data into a 'training' and a 'test' set
	* for each 'test' record run KNN using the 'training' data
	* Does the result of KNN equal the 'test' record bucket??

### Lecture 17 - Randomizing Test Data

* we implement a function to split the dataset in 2 groups
* we shuffle data to avoid bias `const shuffled = _.shuffle(data);`
* we split our dataset in 2 (train, test) using slice
```
	const testSet = _.slice(shuffled, 0, testCount);
	const trainingSet = _.slice(shuffled, testCount);
```
* we return them `return [testSet,trainingSet];`

### Lecture 18 - Generalizing KNN

* we will run 'runAnalysis()' many times for each row in testSet
* we put knn core in a helper function 'knn()' which will run on 'outputs' aka trainingSet multiple times
* in our recurcive runAnalysis each knn run will get a new predicition point which will be the dropPosition of the testSet row
* so we pass it as param passing it in the 'prediction()'
```
function knn(data, point) {
	 return _.chain(data)
	  	.map(row => [distance(row[0], point),row[3]])
		.sortBy(row => row[0])
		.slice(0,k)
		.countBy(row => row[1])
		.toPairs()
		.sortBy(row => row[1])
		.last()
		.first()
		.parseInt()
		.value();
}

function distance(pointA,pointB) {
	return Math.abs(pointA-pointB);
}
```

### Lecture 19 - Gauging Accuracy

* 