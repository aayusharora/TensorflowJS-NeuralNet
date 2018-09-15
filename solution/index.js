const tf = require('@tensorflow/tfjs');
const iris = require('./iris.json');
const irisTesting = require('./testingIris.json');

// Mapping the trainingdata
const trainingData = tf.tensor2d(iris.map(item=> [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]
),[144,4])

// Mapping the testing data
const testingData = tf.tensor2d(irisTesting.map(item=> [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]
),[3,4])

// creating model
const outputData = tf.tensor2d(iris.map(item => [
    item.species === 'setosa' ? 1 : 0,
    item.species === 'virginica' ? 1 : 0,
    item.species === 'versicolor' ? 1 : 0

]), [144,3])

// Creating Model
const model = tf.sequential();

// Adding Input Layer
model.add(tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 5
}))

// Adding Hidden Layer
model.add(tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 3
}))

// Adding output layer
model.add(tf.layers.dense({
    inputShape: [3],
    activation: "sigmoid",
    units: 3
}))

// compiling model
model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(.06)
})

// predicting model
model.fit(trainingData, outputData, {epochs: 100})
    .then(() => {
        model.predict(testingData).print();
})
