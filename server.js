const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');
const use = require('@tensorflow-models/universal-sentence-encoder');

const app = express();
const port = 3000;

app.use(bodyParser.json());

let model;

async function loadModel() {
  model = await use.load();
  console.log('Model loaded');
}

function cosineSimilarity(vec1, vec2) {
  const dotProduct = tf.sum(vec1.mul(vec2));
  const vec1Mag = tf.sqrt(tf.sum(vec1.square()));
  const vec2Mag = tf.sqrt(tf.sum(vec2.square()));
  return dotProduct.div(vec1Mag.mul(vec2Mag));
}

async function calculateSemanticSimilarity(sentence1, sentence2) {
  if (!model) {
    await loadModel();
  }

  const embeddings = await model.embed([sentence1, sentence2]);
  const similarity = cosineSimilarity(
    embeddings.slice([0, 0], [1]),
    embeddings.slice([1, 0], [1])
  );

  const result = await similarity.data();
  return result[0];
}

app.post('/calculate-similarity', async (req, res) => {
  const { sentence1, sentence2 } = req.body;
  
  if (!sentence1 || !sentence2) {
    return res.status(400).json({ error: 'Both sentences are required' });
  }

  try {
    const similarity = await calculateSemanticSimilarity(sentence1, sentence2);
    res.json({ similarity: similarity.toFixed(4) });
  } catch (error) {
    console.error('Error calculating similarity:', error);
    res.status(500).json({ error: 'An error occurred while calculating similarity' });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
  loadModel(); // Load the model when the server starts
});