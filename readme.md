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

* we assemble all pieces together in the runAnalysis function
* we split the dataset keeping 10 rows as testSet `const [testSet, trainingSet] = splitDataset(outputs, 10);`
* we do a for loop to run knn for each test datarow and we just console log result for now
```
	for (let i = 0; i < testSet.length; i++){
		const bucket = knn(trainingSet,testSet[i][0]);
		console.log(bucket);
	}
```
* we test
* we need to compare knn predictions to the test set actual bucket results. we cl them  `console.log(bucket, testSet[i][3]);` accuracy is poor 

### Lecture 20 - Printing a Report

* we just increase a counter at each correct prediction and we do a cl in the end as report
```
function runAnalysis() {
	const testSetSize = 10;
	const [testSet, trainingSet] = splitDataset(outputs, testSetSize);

	let numberCorrect = 0;
	for (let i = 0; i < testSet.length; i++){
		const bucket = knn(trainingSet,testSet[i][0]);
		if (bucket === testSet[i][3]){
			numberCorrect++;
		}
```

### Lecture 21 - Refactoring Accuracy Reporting

* we refactor  runAnalysis using lodash and .chain()
* we use filter to reduce the array keeping only corect predictions
```
	const accuracy _.chain(testSet)
	 .filter(testPoint => knn(trainingSet, testPoint[0]) === testPoint[3])
	 .size()
	 .divide(testSetSize)
	 .value()
```

### Lecture 22 - Investigating Optimal K Values

* we will wrap ou runAnalisis testvcode in a for loop to test the results for different K vals
* we use lodash .range() instead of for loop)
* we also pass k as parameter at knn()
* we test but we dont see a trend in results
* we change test size to 50 and then to 100 also narrow or widen the k range
* we run k up to 20 and testsize of 100 we also drop balls 1 every pixel and analyze.. we fall in accuracy

### Lecture 23 - Updating KNN for Mutiple Features

* we go to step 2 of our imporvement attempt by adding more feats to the analysis
* we ll modify the algo for multiple variables
* the only change to our algo is that we now havbe to find distance in multiple dimensions (features)
* we will add 1 more feat (bounciness) so we will work in 2d space. distance will be ((x-x0)^2 + (y-y0)^2)^0.5

### Lecture 24 - Multi-Dimensional KNN

* in a 3d space the distance would be ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)^0.5

### Lecture 25 - N-Dimension Distance

* we will use all 3 feats (droppoint, bounciness, ballsize)
* we need to mod the distance method. we will make it able to work for N -dimensions, not just 3
* distance will treat pointA and B as arrays of variable length
* we use lodash and .chain()
* we use .zip() join the 2 arrays as columns
* we use .map to square the diff of 2 nums using array destructutring
* ,sum() all 
* get the value() and square it to 0.5
```
function distance(pointA,pointB) {
	return _.chain(pointA)
			.zip(pointB)
			.map(([a,b])=> (a-b)**2)
			.sum()
			.value()**0.5;
}
```

### Lecture 26 - Arbitrary Feature Spaces

* we mod knn() to be able to pas sin dimension arrays of feats
* our label is always the last element. we use lodash .initial() to get n-1 first for feats and .last() to get last element
* we dont use .pop() from vanilla JS or shift as they mod the array
* we also use .initial() at point to get rid of label so that araays in distance match in length and zip() works
* this creates a problem in the future when we will want to use the model to do predictions (we wont have a label in the pointso it will fail)
```
	 return _.chain(data)
	  	.map(row => {
	  		return [distance(_.initial(row), point), _.last(row)];
	  	})
```
* so we should not use iniital(point) but manually clear out the array in the function call `.filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === testPoint[3])`

### Lecture 27 - Magnitude Offsets in Features

* we retest for 10/10 dataset. now it takes a long time
* if we plot the points in real scale we see that there is no actual variation in bounciness. so we wont get a good indication from this feat. same for ball size as distances are squared
* this is solved with normalizing feats

### Lecture 28 - Feature Normaization

* we can normalize or standardize our data
* normaization: divide by max val so that tange is 0-1
* standarization: find the standard deviation and move the 0 of our range to this point
* what i get in standarization is a normal distribution around 0 (bell curve)
* for normalization we use MinMax method: Normalized DataSet = (FeatureVal - minOfFeatureVals)/(maxOfFeatureVals - minOfFeatureVals)
* we apply normization one feat at a time
* we test it in JSPlayground using lodash
* Normalization dramaticaly improves KNN

### Lecture 29 - Normalization with MinMax

* we add a new func passing in the data nad the num of feats aka columns we want to normalize `minMax(data, featureCount)`
* label should not get normalized
* we iterate through columns we extract them with .map()
* we get minn and max of column with lodash
* we iterating thriugh column applying minMax to each element
```

function minMax(data, featureCount) {
	const clonedData = _.cloneDeep(data);

	for(let i=0;i<featureCount;i++){
		const column = clonedData.map(row=>row[i]);
		const min = _.min(column);
		const max = _.max(column);
		for(let j=0;j<column.length;j++){
			clonedData[j][i] = (clonedData[j][i] -min) / (max -min);
		}
	}

	return clonedData;
}
```

### Lecture 30 - Applying Normalization

* we test it in console and apply it in runAnalysis() in the testrtrain split `const [testSet, trainingSet] = splitDataset(minMax(outputs, 3), testSetSize);`
* we test

### Lecture 31 - Feature Selection with KNN

* our results are bad even after normalizing
* ou intution says that:
	* Changes to Drop Position: Predictable changes to Output
	* Changes to Bounciness: Changes our output, but not predictably
* In Python plotting the correlation in a scatterplot would show all these
* We test the game playing with bounciness. we see that it has a detrimental effect to the analysis. we might get better ignoring it.
* Selecting features based on the corelation with label is Feature Selection
* if we dont have tools to prove the correlation (eg Python) we can run KNN for each feature and see the results (accuracy)

### Lecture 32 - Objective Feature Picking

* we mod runAnalysis()
* we will fix k and select a column (feat)
* we limit  range(0,3) so 0 1 2 to use it for column index. we will hardcode k
* we extract feature column from data and label `const data = _.map(outputs,row=> [row[feature], _.last(row)]);`
* we move tranitestsplit in analysis passing the new dataset `const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize);`
* we also make knn param for label parametrical `testPoint => knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint)` 

### Lecture 33 - Evaluating Different Feature Values

* we see that indeed drop position has the largest effect in KNN
* the other 2 affect the result but cannot help us predict the result

## Section 3 - Onwards to Tensorflow JS!

### Lecture 34 - Let's Get our Bearings

* Key points from our very frugal into to ML:
	* Features vs Labels
	* Test vs Train sets of data
	* Feature Normalization
	* Common data structures (nested arrays)
	* Feature Selection
* Lodash:
* Pros: 
	* methods for just about everything we need
	* Excellent API desing (e.g. .chain())
	* Skills trasferable to other JS projects
* Cons: 
	* Extremely slow (relatively)
	* Not 'numbers' focused
	* Some things are awkward (getting columns of values)
* Tesorflow JS:
* Pros:
	* Similar API to Lodash
	Extremely Fast for numeric calcuilations
	* Has a 'low-level' linear algebra API + higher level API for ML
	* Similar API to numpy (popular Python numerical lib)
* Cons: 
	* still in active development

### Lecture 35 - A Plan to Move Forward

* Plan on tackling Tensorflow JS
	* Learn some fundamentals around Tensorflow JS
	* Go through a couple of exercises with Tensorflow
	* Rebuild KNN algorithm using Tensorflow
	* Build other algorithms with Tensorflow
* Rembember that:
	* The fastest way to learn ML is to master fundamental operations around working with Data
	* Strong knowledge of data handling basics makes applying any algorithm trivial

### Lecture 36 - Tensor Shape and Dimension

* [Tensorflow JS Site](https://js.tensorflow.org/)
* Tensorflow #1 Job when you begin ML is to make working with numbers in nested arrays really easy
* The core unit in a Tensorflow program is the Tensor (A JS object that wraps a collection of numbers structured in arrays)
* A Tensor in a program language agnostic definition is a multidimensional vector
* A core property of Tensors is Dimensions (like normal arrays in any language or tables in linear algebra)
* An easy way to tell the dimensions of a Tensor is to count the opening square braces
* Linear Algebra knowledge is a MUST
* Another core property of a Tensor is Shape: How many records (elements) in each dimension AKA size of array or Table (for JS remember .length on each dimension from outside in)
* e.g [[5,10,17],[18,4,2].length=3].length=2 => Shape [2,3]
* 2D is the most important dimension we will work with. for @D shape is [#rows, #columns]
* shapes are always in brackets. even for 1d

### Lecture 37  - Elementwise Operations

* to access the tensorflow library in JSPlayground we use 'tf.'
* to create a tensor instance we use `const data = tf.tensor([1,2,3])`
* tensor comes pack with methods and properties like '.shape' `data.shape` gives [3]
* we create a second tensor `const otherData = tf.tensor([4,5,6])` 
* tensors support linear algebra mathematical operations like 
	* elementwise addition `data.add(otherData)`
	* elementwise subtraction `data.sub(otherData)` => [-3,-3,-3]
	* elementwise multiplication `data.mul(otherData)` => [4,10,18]
	* elementwise division `data.div(otherData)` => [0.25,0.4,0.5]
* elementwise operations work on elements of same index and the resuilt is in same index  in a new tensor
* elementwise operations can be comparative or logical
* if we call `data` in still [1,2,3] do elementwoise operations do not mutate operands
* for elementwise operations shapes have to match!!!
* elementwise operations work for multidiventional tensors as well

### Lecture 38 - Broadcasting Operations

* sometimes we can do operations on tensors whose shapes don't match. for elementwise operations like add the  value of the smallest shape is broadcasted to do operatons with the other elements e.g [1,2,3] + [4] = [5,6,7]
* Broadcasting works when taking the shape of  both tensors. from right to left the shapes are equal or one is 1. if they are equal normal operation takes effect. if one has shape 1  in one dimension its value is 'broadcasted to other elements' so that operation can be performed e.g [[1,2,3],[4,5,6]] + [[1],[1]] === [[1,2,3],[4,5,6]] + [[1,1,1],[1,1,1]] = [[2,3,4],[5,6,7]]

* broadcastin is alowed when there is no value in one shape. 
* so last dimension size can match or 1, previous have to match , first can match or non-exist

### Lecture 39 - Logging Tensor Data

* Tensors are JS objects so we cannot just use console.log()
* to see how they look like we use `data.print()` assuming data is a tensor object
* we cannot use `console.log(data.print())`

### Lecture 40 - Tensor Accessors 

* accessors are used for debuging purposes not for actual programs
* we make a tensor `const data = tf.tensor([10,20,30]);`
* we can access specific element giving their index e.g `data.get(0)` returns 10
* Tensors are NOT arrays. we cannot use data[0] for a multidimensional Tensor we add arguments in get(). 
* get dimensions must match get argument count
* there is no .set() method. we cannot set specific elements

### Lecture 41 - Creating Slices of Data

* We can access multiple slices of data in a tensor. no need for lodash hacking
* we add a sizable tensor
```
const data = tf.tensor([
  [10,20,30],	
  [40,50,60],	
  [10,20,30],	
  [40,50,60],	
  [10,20,30],	
  [40,50,60],		
  [10,20,30],	
  [40,50,60]
]);
```
* if we want to extract center column we use .slice() passing in the starting index and the size. 
* for our example start index is [0,1] 
* size values are not 0 indexed they are 1 based (num of elements) 
* for our example size is [8,1] : 8 rows 1 column `data.slice([0,1],[8,1])`
* if we dont want to hardcode the row count we can use `data.shape[0]` 
* data.shape is  an array
* an other way is to use -1 meaning all `data.slice([0,1],[-1,1])` or starting index to the end

### Lecture 42 - Tensor Concatenation

* to join together tensors we use .concat()
```
const tensorA = tf.tensor([
	[10,20,30],	
  [40,50,60]
]);
const tensorB = tf.tensor([
	[70,80,90],	
  [100,110,120]
]);
tensorA.concat(tensorB);
```
* the result is [[10 , 20 , 30 ], [40 , 50 , 60 ], [70 , 80 , 90 ], [100, 110, 120]]  
* concat by default works across the first dimension (row in our example) [2,3] concat [2,3] is [4,3]
* if we want to concat along a specific dimension we have to spec the dimension index get it from shape) as second argument. so to concat along columns  `tensorA.concat(tensorB,1);` results in [[10, 20, 30, 70 , 80 , 90 ], [40, 50, 60, 100, 110, 120]] or shape [2,6]
* so default is 0
* the parameter is called axis of concatenation (Sounds like Python)

### Lecture 43 - Summing Values Along an Axis

* we showcase an example. we create 2 tensors
```
const jumpData = tf.tensor([
	[70,70,70],
  [70,70,70],	
	[70,70,70],	
	[70,70,70]
]);

const playerData = tf.tensor([
	[1,170],
  [2,170],	
	[3,170],	
	[4,170]
]);
```
* first has jumps for a player . one player per row
* we want to sum the jumps and concat the result on the player data tensor
* calling `jumpData.sum()` sums all elements up. we dont want that
* to sum along an axis we use `jumpData.sum(1)` as we want to sum along the column direction
* we get [210, 210, 210, 210] so our result is transformed to 1D
* we cannot directly concat to playerData. we need to reshape

### Lecture 44 - Massaging Dimensions with ExpandDims

* we need to trasform our sum to [[210],[210],[210],[210]] so from [4] to [4,1]
* sum results in Dimension Reduction!!!!
* concat needs identical dimensions
* to Avoid Dimension Reduction and keep original dimensions in a sum we use a second argument `jumpData.sum(1,true)` 
* now i can concat along the y axis (1) `jumpData.sum(1,true).concat(playerData,1)`
* another way which is tensorflow standard of ading dimensions is  using .expandDims() 
* expandDims accepts an axis on which the expand happens
* expandDims(0) expands the dimensions of the tensor by 1 on the x axis  so [4] => [1,4]
* expandDims(1) expands the dimensions of the tensor by 1 on the y axis  so [4] => [4,1]
* for our example `jumpData.sum(1).expandDims(1).concat(playerData,1)` solves our puzzle

## Section 4 - Applications of Tensorflow

### Lecture 45 - KNN with Regression

*  Steps to Follow:
	* Apply a slightly different KNN algorithm in the browser with Tensorflow JS and fake data
	* Move KNN algorithm to our code editor with real data and run in NodeJS environment
	* Do some optimization
* OUr next example will have to do with house prices
* we will have alist of properties with their location and price
* THe main difference from previous problem in the type of problem
	* drop ball. which bucket? => discrete label => classification
	* location + feats. price of house? linear label => regression
* Our current approach of KNN Algo
	* Find distance between features and prediction point
	* Sort from lowest point to greatest
	* Take the top K records
	* Average the label value of those top K records

### Lecture 46 - A Change in Data Structure

* our data will be fake dataset of 2 feats: longitute + latitude. the label will be a house price
* in the current approach we will split the dataset in features and labels (Python SKlearn style). so we will have 2 tensors
* the rational is that we will do tensorwide operations

### Lecture 47 - KNN with Tensorflow

* we test in JSPlaygrounds
* we set a fkae data set as tensors
```
const features = tf.tensor([
	[-121, 47],
  	[-121.2, 46.5],
	[-122, 46.4],
	[-120.9, 46.7]
]);


const labels = tf.tensor([
	[200],
    [250],
	[215],
	[240]
]);
```
* index of tensors matters as it maps feats to label
* we pass in a prediction point as tensor `const predictionPoint = tf.tensor([-121,46]);`
* we will write KNN in tensorflow.... using barebones linear algebra not bult in algos
* our KNN is 2D so we have 2D distance calc
* first we find distance from pred point (in 1d) using sub() and broadcasting `features.sub(predictionPoint)`
* the we need to square the diffs (square each eleemnt) using '.pow(2)' on the tensor (we use chaining)
* then we need to sum on the y axis. so we chain `.sum(1)`
* then we need to get the root 2  we chain `.pow(0.5)`

### Lecture 48 - Maintaining Order Relationships

* our next step is to sort the results (distances) from lower to greatest.
* shuffling the feats tensor breaks our index link to the labels tensor
* also tensors cannot be sorted
* to solve indexing we concat features wit labels
* first we need to solve dimensioning issue as duming reduces dimenstios
* we fix features dimensioning making it [4,1] with `	.expandDims(1)`
* then we concat along y axis labels `.concat(labels,1)`

### Lecture 49 - Sorting Tensors

* to sort we will use tensor method 'unstack()' which splits the tensor into an array of tensors along the specified axis
* then we can sort using JS
* chaining '.unstack()' splits the tensor into an array of tensor along the x axis (rows). so 1 tensor per row.
* we can access the each row tensor giving the row index `.unstack()[i]`
* all JS arrays come with the sort() method inbuilt
* e.g. 
```
const letters = ['b','a','d','c'];
letters.sort() // ['a', 'b', 'c' 'd']
```
* JS cannot sort tensors or even objects. No error. no result
* we need to pass in sort() a callback to tell it how to sort
* the callback gets 2 arguments say (a, b) these can be any array element do the comparizon and return 1 or -1 
	* 1 means a > b
	* -1 means b > a
* e.g
```
const distances = [
	{ value: 20},
	{ value: 30},
	{ value: 5},
	{ value: 10},
];

distances.sort((a,b) => {
	return a.value > b.value ? 1 : -1;
});
```
* for our problem we can access the 1st element of each tensor with its index and .get() so `.get(0)`.
* our sorting method that we chain becomes
```
	.sort((a,b)=>{
		return a.get(0) > b.get(0) ? 1 : -1;
	})
```

### Lecture 50 - Averaging Top Values

* to take the top k records we use '.slice()'
* note that after using unstack() in our chain we work with vanilla JS arrays so we use vanilla JS array slice().
* it uses a start point and the num of elements `.slice(0,k)`
* to get the average we sum vals together and use '.reduce()' to do it
* JS arrat reduce takes in a callback with 2 args. the  accumulator and the array element. after the callback it takes the accumulators init val. it iterates through the array moding the sum at our will
* to get the average val
```
.reduce((acc,tensor)=>{
  	return acc + tensor.get(1);
	},0)/k
```
* remember the hoyse val isa t index 1
* our tensorflow based custom KNN complete
```
features
  .sub(predictionPoint)
  .pow(2)
	.sum(1)
	.pow(.5)
	.expandDims(1)
	.concat(labels,1)
	.unstack()
	.sort((a,b)=>{
		return a.get(0) > b.get(0) ? 1 : -1;
	})
	.slice(0,k)
	.reduce((acc,tensor)=>{
  	return acc + tensor.get(1);
	},0)/k
```

### Lecture 51 - Moving to the Editor

* we go to /MLKits/knn-tf folder
* our code goes to index.js
* we also have a csv with real housing data + column titles 
* we also have a 'load_csv.js' a JS file to load the csv data
* 'pacakge.json' has all the libs in. we just have to 'npm install'

### Lecture 52 - Loading CSV Data

* we start writing our index.js file
* first we import tensorflow
```
require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
```
* thje first import tells tf how to do the calculations. using the GPU or the CPU
* our import will try to do clacs on the cpu (for gpu use `require('@tensorflow/tfjs-node-gpu');`)
* the second import is our actual import  we can use in our program
* next we import the loadCSV js file `const loadCSV = require('./load-csv');`
* we call loadCSV to read the file. we pass in the csv file name and a config object that contains:
	* a suffle option (useful in ML)
	* a splitTest setting the record count for test
	* a 'dataColumns' to import in our dataset for our analysis
	* a 'labelColumns'
* we do destructuring to get the attributes of interest out of the generated object
```
let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
	shuffle: true,
	splitTest: 10,
	dataColumns: ['lat','long'],
	labelColumns: ['price']
});
```
* we cl to test code correctness
* we see that the tfjs build uses only generic CPU feats not harnessing our specific CPU feats that could boost performance
* we gan compile Tesorflow if we want on our CPU to boost performance

### Lecture 53 - Running an Analysis

* we cp all knn tf code from textbook in a tnn() function
* knn's signature is `function knn(features, labels, predictionPoint, k){}`
* we want to test the method. but all our sample data are plain arrays we have to turn them to tensors first
```
features = tf.tensor(features);
labels = tf.tensor(labels);
testFeatures = tf.tensor(testFeatures);
testLabels = tf.tensor(testLabels);
```
* we call our method for a testpoint and print our prediction
```
const result = knn(features,labels,testFeatures.slice([0,0],[1,-1]),10);
console.log(`Guess: ${result} Actual: ${testLabels.get(0,0)}`);
```
* our prediction is of as we take int oaccount only the location

### Lecture 54 - Reporting Error Percentages

* we will calucalte and report the error
* error = ((expected value) - (predicted value)) / (expected value)
* `error = (testLabels.get(0,0) - result) / testLabels.get(0,0);`
* we decide to work with testdata as arrays to iterate through testset printing the eror
```
features = tf.tensor(features);
labels = tf.tensor(labels);

const result = knn(features,labels,tf.tensor(testFeatures[0]),10);
const error = (testLabels[0][0]- result) / testLabels[0][0];
console.log(`Guess: ${result} Actual: ${testLabels[0][0]}`);
console.log(`Error: ${error * 100 }%`);
```
* we loop through the whole testSet with forEach()
```
testFeatures.forEach((testPoint, index)=>{
	const result = knn(features,labels,tf.tensor(testPoint),10);
	const error = (testLabels[index][0]- result) / testLabels[index][0];
	console.log(`Guess: ${result} Actual: ${testLabels[index][0]} Error: ${error * 100 }%`);
});
```
* we are almost consantly guesing below the actual

### Lecture 55 - Normalization or Standardization

* we need to include other feats, like size
* we mod our loadCsv cofig `dataColumns: ['lat','long', 'sqft_lot'],`
* knn is dimension agnostic so we rerun test. 
* our results improve but are not optimal... we will do normalization
* in visual code we can enable excel viewer to view csv data
* surface varies much mush more than the location so it has a much larger contibution to knn
* normalization or standarization? our surface vals have a normal distribution with some edge cases... no even distribution. so is a good candidate for standarization
* standarization is not affected by edge cases that spoil our metrics

### Lecture 56 - Numerical Standarization in Tensorflow

* the formula of standarization is: (Value - Average)/StandardDeviation
* standarization is applied per column
* we use 'tf.moments()' an inbuilt method passing the tesnor to calculate it on. this method returns an object which among other contains 'mean' AKA average and 'variance'.
* variance is very closely related sto stdDev as stdDev = sqrt(variance)
* tf.moments() works dimilarly to sum. it needs an axis. other wise it works on all datapoints
* an example of doing standarization on a sample array using tf
```
const numbers = tf.tensor([
	[1,2],
	[3,4],
	[5,6]
]);
const {mean, variance } = tf.moments(numbers,0)
numbers.sub(mean).div(variance.pow(.5))
```

### Lecture 57 - Applying Standarization

* we will add standarization in knn()
* we get the mean and varianve aof all used feats in knn `const {mean, variance } = tf.moments(features,0);`
* we apply standarization in predictionPoint `const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(.5))`
* also we start our knn chain by standardizing feats
* also we need to pass our prediction point as tensor....
* our results are still off

### Lecture 58 - Debugging Calculations

* we will use node debugger and chrome debugging tools
* we run `node --inspect-brk index.js` and open chrome broweser
* we navigate to 'chrome://inspect'
* we see the remote process (index.js). we click inspect and see odevtools debugger
* we place a breakpoint in knn() and run code
* we console log features (shape and print) and look oc. we look at our scaled prediction
* it looks ok (close to 0)
* we cp code in console chaining .print() and look at iteration to spot abnormal vals

### Lecture 59 - What Now?

* to improve our algo we can do other steps like check different k vals . or ad dmore feats to the analyzis
* we add 'sqft_living' to feats. imrpovement is much better

## Section 5 - Getting Started with Gradient Descent

### Lecture 60 - Linear Regression

* Linear Regression Pros:
	* Its Fast. Only train once then use it for any prediction
	* Uses methods that will be very important in more complicated ML
* Linear Regression Cons:
	* Lot harder to understant with just intuition

### Lecture 61 - Why Linear Regression?

* 