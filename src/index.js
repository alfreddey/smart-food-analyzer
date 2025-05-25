/*
 * train_food101_mobilenet.js
 *
 * Transfer learning with MobileNet on Food-101 dataset using TensorFlow.js (Node.js)
 *
 * Usage: node train_food101_mobilenet.js
 *
 * Assumes Food-101 images are in `./food-101/images/<class>/*.jpg`
 */

const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

// CONFIGURATION
const DATA_DIR = path.join(__dirname, "food-101", "images");
const IMAGE_SIZE = 224;
const BATCH_SIZE = 8;
const EPOCHS = 5;

// Utility: get class names from subdirectories
function getClassNames(dataDir) {
  return fs
    .readdirSync(dataDir)
    .filter((name) => fs.statSync(path.join(dataDir, name)).isDirectory());
}

// Utility: load file paths and labels
function loadImagePaths(dataDir, classNames) {
  const imagePaths = [];
  classNames.forEach((cls, idx) => {
    const clsDir = path.join(dataDir, cls);
    fs.readdirSync(clsDir).forEach((file) => {
      if (file.match(/\.(jpe?g|png)$/i)) {
        imagePaths.push({ path: path.join(clsDir, file), label: idx });
      }
    });
  });
  return imagePaths;
}

// Create tf.data.Dataset from generator
function createDataset(imagePaths, numClasses) {
  const ds = tf.data.generator(function* () {
    for (const { path: imgPath, label } of imagePaths) {
      const buffer = fs.readFileSync(imgPath);
      let img = tf.node.decodeImage(buffer, 3);
      img = tf.image.resizeBilinear(img, [IMAGE_SIZE, IMAGE_SIZE]);
      img = img.div(255);
      const lbl = tf.oneHot(label, numClasses);
      yield { xs: img, ys: lbl };
    }
  });
  return ds.shuffle(imagePaths.length).batch(BATCH_SIZE);
}

(async () => {
  // 1. Prepare data
  const classNames = getClassNames(DATA_DIR);
  const numClasses = classNames.length;
  console.log(`Classes: ${numClasses}`);

  const allImages = loadImagePaths(DATA_DIR, classNames);
  const split = Math.floor(allImages.length * 0.8);
  const trainPaths = allImages.slice(0, split);
  const valPaths = allImages.slice(split);

  const trainDs = createDataset(trainPaths, numClasses);
  const valDs = createDataset(valPaths, numClasses);

  // 2. Load base MobileNet model
  console.log("Loading MobileNet v1...");
  const mobilenetUrl =
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json";
  const baseModel = await tf.loadLayersModel(mobilenetUrl);
  console.log("MobileNet loaded.");

  // 3. Freeze all base layers
  baseModel.layers.forEach((layer) => (layer.trainable = false));

  // 4. Build transfer model
  const x = baseModel.getLayer("conv_pw_13_relu").output;
  const globalPool = tf.layers
    .globalAveragePooling2d({ dataFormat: "channelsLast" })
    .apply(x);
  const predictions = tf.layers
    .dense({ units: numClasses, activation: "softmax" })
    .apply(globalPool);
  const model = tf.model({ inputs: baseModel.inputs, outputs: predictions });

  model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Print model summary
  model.summary();

  // 5. Train the model
  console.log("Starting training...");
  await model.fitDataset(trainDs, {
    epochs: EPOCHS,
    validationData: valDs,
    callbacks: [tf.node.tensorBoard("./logs")],
  });

  // 6. Save the trained model
  const savePath = "file://" + path.join(__dirname, "food101_mobilenet_model");
  await model.save(savePath);
  console.log(`Model saved to ${savePath}`);
})();
