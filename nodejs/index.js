const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');
const { Tokenizer } = require('./tokenizer');

const MAX_LENGTH = 100;

const tfjsDir = path.join(__dirname, '..', 'python/src/data/tfjs');
const tokenizerFile = path.join(tfjsDir, 'tokenizer.json');
const modelFile =
  'https://raw.githubusercontent.com/quanttb/toxic-comment-classification/dev/python/src/data/tfjs/model.json';

const predict = async (text) => {
  fs.readFile(tokenizerFile, async (error, data) => {
    if (error) {
      console.error(error);
      return;
    }

    const tokenizer = new Tokenizer({
      ...JSON.parse(data).config,
    });

    const tokenizedTestList = tokenizer.textsToSequences([text.trim()]);

    const predictList = [];

    for (let text of tokenizedTestList) {
      const zeroArrLength = MAX_LENGTH - text.length;
      predictList.push(new Array(zeroArrLength).fill(0).concat(text));
    }

    const model = await tf.loadLayersModel(modelFile);

    const result = model.predict(tf.tensor(predictList), { batchSize: 1024 });

    console.log(
      '["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]'
    );
    console.log(result.print());
  });
};

// Must be cleaned before predict
predict(
  'never get head round people take perfectly functional user friendly app destroy suppose update cannot even top dam thing anymore what is point crap crap crap ee ashamed '
);
