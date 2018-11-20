const tf = require('@tensorflow/tfjs');
const iris = require('./training.json');
const irisTesting = require('./testing.json');

// Mapping the trainingdata
const trainingData = tf.tensor2d(iris.map(item=> [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]
),[130,4])

// Mapping the testing data
const testingData = tf.tensor2d(irisTesting.map(item=> [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]
),[14,4])

// creating model
const outputData = tf.tensor2d(iris.map(item => [
    item.species === 'setosa' ? 1 : 0,
    item.species === 'virginica' ? 1 : 0,
    item.species === 'versicolor' ? 1 : 0

]), [130,3])

// Creating Model
const model = tf.sequential();

// Adding Input Layer
model.add(tf.layers.dense({
    inputShape: [4],
    activation: "relu",
    units: 5
}))

// Adding Hidden Layer
model.add(tf.layers.dense({
    inputShape: [5],
    activation: "relu",
    units: 3
}))

// Adding output layer
model.add(tf.layers.dense({
    inputShape: [3],
    activation: "relu",
    units: 3
}))

// compiling model
model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam()
})

async function train_data(){
    console.log('......Loss History.......');
    for(let i=0;i<10;i++){
     let res = await model.fit(trainingData, outputData, {epochs: 5000});
     console.log(res.history.loss[0]);
  }
}

async function main() {
    await train_data();
    console.log('....Model Prediction .....')
    model.predict(testingData).print();
}

main();
  
